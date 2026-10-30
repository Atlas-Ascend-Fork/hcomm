/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../hccl_api_base_test.h"
#include "hcomm/hcomm_res_expt.h"
#include "hcomm_c_adpt.h"
#include "comm_engine/thread/thread_pub.h"
#include "common/result_defs_pub.h"
#include "cpu_thread.h" // CpuThread：断言 CPU + THREAD_TYPE_CPU 分派命中该类型
#include "local_notify_impl.h"
#include "comm_engine/aicpu_launch_manager_pub.h"
#include "aicpu_thread_launch_mgr.h"
#include "aicpu_ts_thread.h"
#include "llt_hccl_stub_rank_graph.h"
#include "launch_aicpu.h"
#include "orion_adapter_rts.h"

using Hccl::HrtGetDevice;

// stub for AicpuThreadLaunchMgr::ThreadKernelLaunchForComm，校验调用方已预分配非零 Entity 句柄
static HcclResult StubThreadKernelLaunchForComm(
    std::vector<std::shared_ptr<hccl::Thread>>& newThreads, const std::string& commId,
    std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle)
{
    for (size_t i = 0; i < newThreads.size(); ++i) {
        EXPECT_NE(aicpuHandle[i], 0);
    }
    return HCCL_SUCCESS;
}

// stub for AicpuThreadLaunchMgr::ThreadKernelLaunchForBase：commId 为空的 alloc 走该分支，
// 真实实现不回拷输出句柄，调用方须先预分配非零 Entity 句柄
static HcclResult StubThreadKernelLaunchForBase(
    std::vector<std::shared_ptr<hccl::Thread>>& newThreads, std::unique_ptr<ThreadHandle[]>& aicpuHandle,
    aclrtBinHandle binHandle)
{
    for (size_t i = 0; i < newThreads.size(); ++i) {
        EXPECT_NE(aicpuHandle[i], 0);
    }
    return HCCL_SUCCESS;
}

// Spy 线程：通过构造/析构计数观测 host 侧线程对象的生命周期，
// 用于验证 FreeThreads 先销毁 device 侧线程、再释放 host 侧 SQ/CQ 资源的顺序
class SpyThreadForFree : public hccl::Thread {
public:
    SpyThreadForFree() { ++aliveCount; }
    ~SpyThreadForFree() override { --aliveCount; }
    static int aliveCount;

    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult DeInit() override { return HCCL_SUCCESS; }
    std::string& GetUniqueId() override { return uniqueId_; }
    uint32_t GetNotifyNum() const override { return 0; }
    size_t GetNotifySize() const override { return 0; }
    LocalNotify* GetNotify(uint32_t index) const override { return nullptr; }
    HcclResult SupplementNotify(uint32_t notifyNum) override { return HCCL_SUCCESS; }
    bool IsDeviceA5() const override { return false; }
    Stream* GetStream() const override { return nullptr; }
    void* GetStreamLitePtr() const override { return nullptr; }
    s32 GetStreamId() const override { return -1; }
    void LaunchTask() const override {}
    void TryLaunchTask() const override {}
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const override { return HCCL_SUCCESS; }
    HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const override { return HCCL_SUCCESS; }
    HcclResult LocalReduce(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp) const override
    {
        return HCCL_SUCCESS;
    }
    bool GetMaster() const override { return false; }
    void SetIsMaster(bool isMaster) override {}

private:
    std::string uniqueId_{"spy_thread"};
};
int SpyThreadForFree::aliveCount = 0;

static bool g_destroyCalled = false;
static int g_aliveAtDestroy = -1;
static ThreadHandle g_destroyDeviceHandle = 0;
static uint32_t g_destroyListNum = 0;

static HcclResult StubThreadKernelLaunchDestroy(ThreadHandle* threadHandles, uint32_t listNum, aclrtBinHandle binHandle)
{
    (void)binHandle;
    g_destroyCalled = true;
    g_aliveAtDestroy = SpyThreadForFree::aliveCount;
    g_destroyListNum = listNum;
    g_destroyDeviceHandle = (threadHandles != nullptr && listNum > 0) ? threadHandles[0] : 0;
    return HCCL_SUCCESS;
}

class TestHcclThread : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        unsetenv("HCCL_DFS_CONFIG");
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

    void AllocCpuTsThreads(uint32_t threadNum, uint32_t notifyNum, ThreadHandle* threads)
    {
        bool isDeviceSide{false};
        MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
        std::vector<uint32_t> notifyNums(threadNum, notifyNum);
        HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, threadNum, notifyNums.data(), threads);
        ASSERT_EQ(ret, HCCL_SUCCESS);
        for (uint32_t i = 0; i < threadNum; ++i) {
            EXPECT_EQ(reinterpret_cast<Thread*>(threads[i])->GetNotifyNum(), notifyNum);
        }
    }
};

TEST_F(TestHcclThread, Ut_NotifyLoadType_When_AicpuEngine_Expect_HostNotify)
{
    NotifyLoadType notifyLoadType = NotifyLoadType::DEVICE_NOTIFY;
    EXPECT_EQ(CommEngineToNotifyLoadType(COMM_ENGINE_AICPU_TS, notifyLoadType), HCCL_SUCCESS);
    EXPECT_EQ(notifyLoadType, NotifyLoadType::HOST_NOTIFY);

    notifyLoadType = NotifyLoadType::DEVICE_NOTIFY;
    EXPECT_EQ(GetNotifyLoadType(COMM_ENGINE_AICPU, THREAD_TYPE_TS, notifyLoadType), HCCL_SUCCESS);
    EXPECT_EQ(notifyLoadType, NotifyLoadType::HOST_NOTIFY);
}

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateHostCpuTsCommEngineThread_Return_HCCL_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    std::shared_ptr<Thread> cpuHandle;
    HcclResult ret = CreateThread(
        COMM_ENGINE_CPU_TS, THREAD_TYPE_TS, StreamType::STREAM_TYPE_ONLINE, 3, NotifyLoadType::HOST_NOTIFY, cpuHandle);
    EXPECT_EQ(ret, 0);
    ret = cpuHandle->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint64_t cpu = reinterpret_cast<ThreadHandle>(cpuHandle.get());
    Stream* stream = GetStream(cpu);
    EXPECT_NE(stream, nullptr);
    void* notify = GetNotify(cpu, 0);
    EXPECT_NE(nullptr, notify);
}

// HOST CPU 线程（HostDpu kernel）走独立分派：不建 TS 流/notify，只构造薄壳 CpuThread
TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateHostCpuThreadType_Return_CpuThread)
{
    std::shared_ptr<Thread> handle;
    HcclResult ret = CreateThread(
        COMM_ENGINE_CPU, THREAD_TYPE_CPU, StreamType::STREAM_TYPE_ONLINE, 0, NotifyLoadType::HOST_NOTIFY, handle);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    ASSERT_NE(nullptr, handle);
    EXPECT_NE(nullptr, std::dynamic_pointer_cast<CpuThread>(handle));
    // 与 CPU_TS 的 CpuTsThread 分派互斥：threadType 决定是否落到 dpu 实体
    std::shared_ptr<Thread> tsHandle;
    EXPECT_EQ(
        HCCL_SUCCESS,
        CreateThread(
            COMM_ENGINE_CPU, THREAD_TYPE_TS, StreamType::STREAM_TYPE_ONLINE, 0, NotifyLoadType::HOST_NOTIFY, tsHandle));
    ASSERT_NE(nullptr, tsHandle);
    EXPECT_EQ(nullptr, std::dynamic_pointer_cast<CpuThread>(tsHandle));
}

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateAicpuTsCommEngineThread_Return_HCCL_Success)
{
    std::shared_ptr<Thread> aicpuHandle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = CreateThread(
        COMM_ENGINE_AICPU_TS, THREAD_TYPE_TS, StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::HOST_NOTIFY,
        aicpuHandle);
    EXPECT_EQ(ret, 0);
    ret = aicpuHandle->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint64_t aicpu = reinterpret_cast<ThreadHandle>(aicpuHandle.get());
    Stream* aicpuStream = GetStream(aicpu);
    EXPECT_NE(aicpuStream, nullptr);
    void* aicpuNotify = GetNotify(aicpu, 0);
    EXPECT_NE(nullptr, aicpuNotify);
}

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateNotSurportCommEngineThread_Return_HCCL_E_NOT_SUPPORT)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = CreateThread(
        COMM_ENGINE_RESERVED, THREAD_TYPE_TS, StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY, Handle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestHcclThread, UT_When_DeviceSide_ResourceAllocateFail_expect_return_HcclEInternal)
{
    bool isDeviceSide{true};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    uint32_t notifyNum[2] = {3, 3};
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, notifyNum, thread);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_ThreadIsNullptr_Allocate_expect_Return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    uint64_t* thread = nullptr;
    uint32_t notifyNum[2] = {3, 3};
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, notifyNum, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_WithUnsupportedEngine_expect_return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    uint32_t notifyNum[2] = {3, 3};
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AIV, 2, notifyNum, thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_Thread_Allocate_0Num_expect_Return_HCCL_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    uint32_t notifyNum[2] = {3, 3};
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AIV, 0, notifyNum, thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_WithNotifyInitFail_expect_return_HcclERuntime)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtNotifyGetOffset).stubs().will(returnValue(HCCL_E_RUNTIME));
    ThreadHandle thread[3];
    uint32_t notifyNum[2] = {3, 3};
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, notifyNum, thread);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_CpuTsThread_Allocate_expect_Return_HCCL_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    uint32_t notifyNum[2] = {3, 3};
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, notifyNum, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);
    Thread* threadptr1 = reinterpret_cast<Thread*>(thread[1]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);
    EXPECT_EQ(threadptr1->GetNotifyNum(), 3);
    ret = HcommThreadFree(thread, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_ThreadNum_Is_0_expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 0;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 0, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 规格放开回归：每线程 notify 超出旧上限 64 后在 CPU 引擎应成功（上限由底层资源决定）
TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_Notify_Exceeds_Old_Limit_expect_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = hccl::HCOMM_NOTIFY_MAX_NUM + 1;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommThreadFree(&thread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 流创建失败（Stream 构造函数仅打日志不置失败态）：HostInit/Init 须显式拦截空流，
// 接口按 HCCL_E_RUNTIME 报错而非携带坏流句柄返回 SUCCESS
TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_StreamCreateFail_AicpuTs_Expect_ErrorReturn)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceRefresh).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtStreamCreateWithFlags).stubs().will(returnValue(HCCL_E_RUNTIME));

    ThreadHandle thread;
    uint32_t notifyNum = 2;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_StreamCreateFail_CpuTs_Expect_ErrorReturn)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtStreamCreateWithFlags).stubs().will(returnValue(HCCL_E_RUNTIME));

    ThreadHandle thread;
    uint32_t notifyNum = 2;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_expect_Return_HCCL_Success)
{
    ThreadHandle thread[2];
    AllocCpuTsThreads(2, 3, thread);
    HcommResult ret = HcommThreadFree(thread, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_ThreadNum_Is_0_expect_Return_HCCL_E_PARA)
{
    ThreadHandle thread[2];
    AllocCpuTsThreads(2, 3, thread);
    HcommResult ret = HcommThreadFree(thread, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_ThreadNullptr_expect_Return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle* thread = nullptr;
    HcommResult ret = HcommThreadFree(thread, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_FreeThreads_When_DeviceThread_Expect_HostReleasedAfterDeviceDestroy)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    g_destroyCalled = false;
    g_aliveAtDestroy = -1;
    g_destroyDeviceHandle = 0;
    g_destroyListNum = 0;
    SpyThreadForFree::aliveCount = 0;

    auto spyThread = std::make_shared<SpyThreadForFree>();
    EXPECT_EQ(SpyThreadForFree::aliveCount, 1);
    ThreadHandle hostHandle = reinterpret_cast<ThreadHandle>(spyThread.get());
    ThreadHandle deviceHandle = hostHandle + 0x1000;

    ASSERT_EQ(SaveThreads(std::vector<std::shared_ptr<Thread>>{spyThread}), HCCL_SUCCESS);
    ASSERT_EQ(FillThreadD2HMap(&deviceHandle, &hostHandle, 1), HCCL_SUCCESS);
    spyThread.reset(); // 释放测试栈引用，仅剩全局映射表持有 host 线程
    EXPECT_EQ(SpyThreadForFree::aliveCount, 1);

    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchDestroy).stubs().will(invoke(StubThreadKernelLaunchDestroy));

    HcclResult ret = FreeThreads(&deviceHandle, 1, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_TRUE(g_destroyCalled);
    EXPECT_EQ(g_destroyListNum, 1U);
    EXPECT_EQ(g_destroyDeviceHandle, deviceHandle);
    EXPECT_EQ(g_aliveAtDestroy, 1);             // device 侧销毁时 host 线程仍存活
    EXPECT_EQ(SpyThreadForFree::aliveCount, 0); // FreeThreads 返回后 host 线程已释放
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_Allocate_WithStream_expect_return_HcclSuccess)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcommResult freeRet = HcommThreadFree(&thread, 1);
    EXPECT_EQ(freeRet, HCCL_SUCCESS);
    delete stream;
}

TEST_F(TestHcclThread, UT_TestHcommThreadFree_When_ThreadNumZero_expect_return_HCCL_E_PARA)
{
    ThreadHandle thread = 0;
    HcommResult ret = HcommThreadFree(&thread, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, UT_TestHcommThreadFree_When_Nullptr_expect_return_HCCL_E_PTR)
{
    HcommResult ret = HcommThreadFree(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, UT_TestHcommThreadFree_When_HandleNotFound_expect_return_HCCL_E_NOT_FOUND)
{
    ThreadHandle thread = 0x9999;
    HcommResult ret = HcommThreadFree(&thread, 1);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_ThreadNullptr_expect_return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle* thread{nullptr};
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_WithInvalidEngine_expect_return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_AICPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
    delete stream;
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_NotifyInitFailed_expect_return_HCCL_E_RUNTIME)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtNotifyGetOffset).stubs().will(returnValue(HCCL_E_RUNTIME));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

/* =========== HcommThreadResGetInfo (L0 内部 4 参数版) =========== */

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_Normal_Expect_Return_HCCL_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 3;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* info = nullptr;
    ret = HcommThreadResGetInfo(thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &info);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(info, nullptr);

    HcommThreadFree(&thread, 1);
}

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_InfoNull_Expect_Return_HCCL_E_PTR)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 3;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommThreadResGetInfo(thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    HcommThreadFree(&thread, 1);
}

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_ThreadZero_Expect_Return_HCCL_E_PTR)
{
    void* info = nullptr;
    HcommResult ret
        = HcommThreadResGetInfo(0, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &info);
    EXPECT_EQ(ret, HCOMM_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_ResTypeNotSupport_Expect_Return_HCCL_E_PARA)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 3;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* info = nullptr;
    ret = HcommThreadResGetInfo(thread, ThreadResType::THREAD_RES_TYPE_INVALID, sizeof(ThreadResTypeStream), &info);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcommThreadFree(&thread, 1);
}

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_InfoLenMismatch_Expect_Return_HCCL_E_PARA)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 3;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* info = nullptr;
    ret = HcommThreadResGetInfo(thread, ThreadResType::THREAD_RES_TYPE_STREAM, 1, &info);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcommThreadFree(&thread, 1);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_CpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_AicpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    // mock 需填充句柄：L1 SupplementThread 基于句柄查询 stream 信息，不填会拿到 0 导致失败
    // 用例 commName 为空串，alloc 经 StoreThreadHandles 走 commId == "" 的 ForBase 分支（真实运行时 commId
    // 非空走 ForComm）；ForComm 保留 stub 防御 export 等路径触发
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForComm));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForBase).stubs().will(invoke(StubThreadKernelLaunchForBase));
    MOCKER_CPP(&HcclCommProfiling::ReportKernel).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
}
TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_AicpuTsThread_Reuse_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    // mock 需填充句柄：L1 SupplementThread 基于句柄查询 stream 信息，不填会拿到 0 导致失败
    // 用例 commName 为空串，alloc 经 StoreThreadHandles 走 commId == "" 的 ForBase 分支（真实运行时 commId
    // 非空走 ForComm）；ForComm 保留 stub 防御 export 等路径触发
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForComm));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForBase).stubs().will(invoke(StubThreadKernelLaunchForBase));
    MOCKER_CPP(&AicpuThreadLaunchMgr::SupplementNotifyKernelLaunch).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&HcclCommProfiling::ReportKernel).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 2, 3, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 3, 3, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 3, 4, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 4, 5, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));

    void* comm = nullptr;
    ThreadHandle thread;
    HcclResult ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_ThreadNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle* thread{nullptr};
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_CollCommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(true));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_Normal_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(notifyNum, 3);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_Normal_Reuse_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadLaunchMgr::SupplementNotifyKernelLaunch).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1; // 非CCU模式，避免拉起CCU平台层
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);
    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 3, 3, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread*>(thread[1]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 3, 4, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread*>(thread[2]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 4);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 4, 5, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread*>(thread[3]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 5);

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(notifyNum, 5);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t notifyNum;
    comm = nullptr;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_notifyNumNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t* notifyNum{nullptr};
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_CollCommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    uint32_t notifyNum;
    HcclResult ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_Acquire_CpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    void* comm = nullptr;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_threadNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle* thread{nullptr};
    void* comm = (void*)0x1234;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_rtStreamNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* rtStream = nullptr;
    ThreadHandle thread;
    void* comm = (void*)0x1234;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_CollcommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_A3_Acquire_CpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    // IsCommunicatorV2() 现为运行时谓词 EqualToOrGreaterThan950()：A3(910B) 需显式固定设备类型，
    // 否则走到 V2 分支（collComm 未初始化→E_PTR）
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireWithStream).stubs().will(returnValue(HCCL_SUCCESS));

    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_MAX_Thread_AicpuTsThread_Return_HCCL_E_UNAVAIL)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireV2).stubs().will(returnValue(HCCL_E_UNAVAIL));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 201, 0, &thread);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_MAX_Notify_AicpuTsThread_Return_HCCL_E_UNAVAIL)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireV2).stubs().will(returnValue(HCCL_E_UNAVAIL));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, HCOMM_THREAD_NOTIFY_MAX_NUM, &thread);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_Normal_Expect_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);

    ThreadResTypeStream stream;
    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_commNull_Return_HCCL_E_PTR)
{
    HcclComm comm = nullptr;
    ThreadHandle thread;
    ThreadResTypeStream stream;
    HcclResult ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_infoNull_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_threadNotFound_Return_HCCL_E_NOT_FOUND)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    // 句柄须显式初始化为非零且未登记的值，避免栈残留 0 命中空指针早退导致用例随机失败
    ThreadHandle thread = 0x9999;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_restypeNotSupport_Return_HCCL_E_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);

    ThreadResTypeStream stream;
    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_INVALID, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_infoLenMismatch_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, 1, &stream);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_ApplyRdmaTrafficAndServiceLevel_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;    // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 120; // 配置RDMA Traffic Class为120，验证正常配置RDMA Traffic Class的场景
    config.hcclRdmaServiceLevel = 3;   // 配置RDMA Service Level为3，验证正常配置RDMA Service Level的场景
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_RdmaServiceLevel_Abnormal_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 8;          // 配置RDMA Service Level超过正常值
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_RdmaTrafficClass_Abnormal_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 300;        // 配置RDMA Traffic Class超过约束范围
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestHcclThread, Ut_ThreadKernelLaunchDestroy)
{
    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

    constexpr u32 threadHandleNum = 1;
    vector<ThreadHandle> deviceHandle(threadHandleNum);
    void* binHandle = nullptr;
    HcclResult ret
        = AicpuThreadLaunchMgr::ThreadKernelLaunchDestroy(deviceHandle.data(), deviceHandle.size(), binHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// —— host 侧变长参数块拼装断言：捕获 hrtMemSyncCopy 三段 H2D 与 kernel context，校验地址/偏移/长度自洽 ——
namespace {
struct UtMemCopyRecord {
    void* dst;       // 目的地址（device 地址值，仅作数值比较）
    const void* src; // 源地址（仅作数值比较，不解引用——host 侧源在函数返回后已析构）
    vector<char> srcCopy; // 源内容深拷贝（src 可能指向被测函数栈上对象，须在调用时捕获内容）
    uint64_t count;
    HcclRtMemcpyKind kind;
};
vector<UtMemCopyRecord> g_utMemCopyRecords;

HcclResult UtMemSyncCopyCapture(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind)
{
    (void)destMax;
    if (kind == HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST) {
        (void)memset(dst, 0, count);
    }
    UtMemCopyRecord rec;
    rec.dst = dst;
    rec.src = src;
    rec.count = count;
    rec.kind = kind;
    rec.srcCopy.assign(static_cast<const char*>(src), static_cast<const char*>(src) + count);
    g_utMemCopyRecords.push_back(std::move(rec));
    return HCCL_SUCCESS;
}

uint64_t g_utKernelCtx = 0;
HcclResult UtAicpuKernelLaunchCapture(
    const rtStream_t stm, void* addr, u32 size, aclrtBinHandle binHandle, const string& kernelName, bool isInitTask,
    u16 timeOut, void* tilingDataPtr, u32 tilingDataSize)
{
    (void)stm;
    (void)size;
    (void)binHandle;
    (void)kernelName;
    (void)isInitTask;
    (void)timeOut;
    (void)tilingDataPtr;
    (void)tilingDataSize;
    g_utKernelCtx = *static_cast<uint64_t*>(addr);
    return HCCL_SUCCESS;
}
} // namespace

TEST_F(TestHcclThread, Ut_ThreadKernelLaunchImpl_When_Normal_Expect_VariableLayoutConsistent)
{
    const string uidA = "UT-uid-thread-A";
    const string uidB = "UT-uid-thread-B-longer";
    vector<shared_ptr<Thread>> newThreads;
    auto threadA = make_shared<AicpuTsThread>(uidA);
    auto threadB = make_shared<AicpuTsThread>(uidB);
    threadA->SetCommEngine(COMM_ENGINE_AICPU);
    threadB->SetCommEngine(COMM_ENGINE_AICPU);
    // threadB 登记 order 控制流标记，覆盖 orderCtrlFlag 两种取值
    MarkOrderCtrlThread(reinterpret_cast<ThreadHandle>(threadB.get()));
    newThreads.emplace_back(threadA);
    newThreads.emplace_back(threadB);

    g_utMemCopyRecords.clear();
    g_utKernelCtx = 0;
    // ThreadKernelLaunchImpl 现在要求调用方先预分配并传入 Entity 句柄。
    unique_ptr<ThreadHandle[]> aicpuHandle = make_unique<ThreadHandle[]>(newThreads.size());
    aicpuHandle[0] = static_cast<ThreadHandle>(0x4000ULL);
    aicpuHandle[1] = static_cast<ThreadHandle>(0x4100ULL);
    MOCKER(hrtMemSyncCopy).stubs().will(invoke(UtMemSyncCopyCapture));
    MOCKER(AicpuAclKernelLaunch).stubs().will(invoke(UtAicpuKernelLaunchCapture));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtStreamSetMode).stubs().will(returnValue(HCCL_SUCCESS));
    // ForBase 路径 needDeviceInfo=true，需 mock 设备信息查询
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    // 经公有入口 ForBase 走完整拼装+下发（ThreadKernelLaunchImpl 为私有实现，产品代码同款路径）
    HcclResult ret = AicpuThreadLaunchMgr::ThreadKernelLaunchForBase(
        newThreads, aicpuHandle, reinterpret_cast<aclrtBinHandle>(0x1));
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 四段 H2D（Entity句柄/定长头/entry表/uid缓冲），无 D2H 回读
    ASSERT_EQ(g_utMemCopyRecords.size(), 4U);
    const UtMemCopyRecord& entityCopy = g_utMemCopyRecords[0];
    const UtMemCopyRecord& hdrCopy = g_utMemCopyRecords[1];
    const UtMemCopyRecord& entryCopy = g_utMemCopyRecords[2];
    const UtMemCopyRecord& uidCopy = g_utMemCopyRecords[3];

    // 第 1 段：调用方预分配的 Entity 句柄数组作为输入下发
    EXPECT_EQ(entityCopy.kind, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(entityCopy.src, static_cast<const void*>(aicpuHandle.get()));
    EXPECT_EQ(entityCopy.count, static_cast<uint64_t>(sizeof(ThreadHandle) * newThreads.size()));
    const ThreadHandle* entityHandles = reinterpret_cast<const ThreadHandle*>(entityCopy.srcCopy.data());
    EXPECT_EQ(entityHandles[0], static_cast<ThreadHandle>(0x4000ULL));
    EXPECT_EQ(entityHandles[1], static_cast<ThreadHandle>(0x4100ULL));

    // 第 2 段：定长头，字段与入参一致
    EXPECT_EQ(hdrCopy.kind, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(hdrCopy.count, static_cast<uint64_t>(sizeof(ThreadMgrAicpuParam)));
    const ThreadMgrAicpuParam* hdr = reinterpret_cast<const ThreadMgrAicpuParam*>(hdrCopy.srcCopy.data());
    EXPECT_EQ(hdr->threadNum, 2U);
    EXPECT_EQ(hdr->engine, COMM_ENGINE_AICPU);
    EXPECT_EQ(hdr->paramBufSize, static_cast<uint64_t>(uidA.size() + uidB.size()));
    EXPECT_EQ(hdr->deviceHandle, entityCopy.dst); // 头部句柄缓冲 == Entity 下发目标

    // 第 3 段：entry 表紧接头之后，地址/长度与头部字段自洽，偏移/长度/orderCtrlFlag 与各 uid 一一对应
    EXPECT_EQ(entryCopy.kind, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(entryCopy.dst, reinterpret_cast<void*>(static_cast<uintptr_t>(hdr->entryTableAddr)));
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(entryCopy.dst),
        reinterpret_cast<uintptr_t>(hdrCopy.dst) + sizeof(ThreadMgrAicpuParam));
    EXPECT_EQ(entryCopy.count, static_cast<uint64_t>(newThreads.size() * sizeof(ThreadParamEntry)));
    const ThreadParamEntry* entries = reinterpret_cast<const ThreadParamEntry*>(entryCopy.srcCopy.data());
    EXPECT_EQ(entries[0].uidOffset, 0U);
    EXPECT_EQ(entries[0].uidLen, static_cast<uint32_t>(uidA.size()));
    EXPECT_EQ(entries[0].orderCtrlFlag, 0U);
    EXPECT_EQ(entries[1].uidOffset, static_cast<uint32_t>(uidA.size()));
    EXPECT_EQ(entries[1].uidLen, static_cast<uint32_t>(uidB.size()));
    EXPECT_EQ(entries[1].orderCtrlFlag, 1U);

    // 第 4 段：变长 uid 缓冲紧接 entry 表之后，内容为各 uid 顺序拼接
    EXPECT_EQ(uidCopy.kind, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(uidCopy.dst, reinterpret_cast<void*>(static_cast<uintptr_t>(hdr->paramBufAddr)));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(uidCopy.dst), reinterpret_cast<uintptr_t>(entryCopy.dst) + entryCopy.count);
    EXPECT_EQ(uidCopy.count, hdr->paramBufSize);
    const string expectUid = uidA + uidB;
    EXPECT_EQ(memcmp(uidCopy.srcCopy.data(), expectUid.data(), expectUid.size()), 0);

    // kernel context == 参数块基址
    EXPECT_EQ(g_utKernelCtx, reinterpret_cast<uint64_t>(hdrCopy.dst));
}

// TEST_F(TestHcclThread, Ut_LaunchNotifyKernel)
// {
//     MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
//     MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
//     MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

//     NotifyMgrAicpuParam opParam;
//     void* binCustomHandle = nullptr;
//     HcclResult ret = AicpuLaunchMgr::LaunchNotifyKernel(opParam, binCustomHandle);
//     EXPECT_EQ(ret, HCCL_SUCCESS);
// }

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_CommNullptr_Return_HCCL_E_PTR)
{
    void* comm = nullptr;
    ThreadHandle thread;
    HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_ThreadNullptr_Return_HCCL_E_PTR)
{
    void* comm = (void*)0x1234;
    ThreadHandle* thread = nullptr;
    HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_UseTypeInvalid_Return_HCCL_E_PARA)
{
    void* comm = (void*)0x1234;
    ThreadHandle thread;
    HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_INVALID, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 专用线程路径（ORDER_LAUNCH_DEVICE）UINT16_MAX 入口边界：65536 应被 HcclDeviceOrderThreadCreate 拦截
TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_DeviceOrderNotifyExceedsUint16Max_Expect_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    ThreadHandle thread = 0;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, UINT16_MAX + 1U, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(thread, 0U);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_SingleComm_RepeatedAcquire_Return_SameThreadAndNotifyPlusTwo)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    ThreadHandle thread1;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 3, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread1, 0ULL);
    Thread* threadPtr1 = reinterpret_cast<Thread*>(thread1);
    EXPECT_EQ(threadPtr1->GetNotifyNum(), 3U);

    ThreadHandle thread2;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 5, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread2, thread1);
    Thread* threadPtr2 = reinterpret_cast<Thread*>(thread2);
    EXPECT_EQ(threadPtr2->GetNotifyNum(), 5U);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_MultiComm_Acquire_Return_DifferentThread)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;

    char commName1[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr1 = make_shared<hccl::hcclComm>(1, 1, commName1);
    HcclCommConfig config1;
    UtInitHcclCommConfig(config1);
    config1.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config1.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config1.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr1->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName1, &config1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm1 = static_cast<HcclComm>(hcclCommPtr1.get());

    char commName2[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr2 = make_shared<hccl::hcclComm>(1, 1, commName2);
    HcclCommConfig config2;
    UtInitHcclCommConfig(config2);
    config2.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config2.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config2.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    ret = hcclCommPtr2->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName2, &config2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm2 = static_cast<HcclComm>(hcclCommPtr2.get());

    ThreadHandle thread1;
    ret = HcclDedicatedThreadAcquire(comm1, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 3, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread1, 0ULL);

    ThreadHandle thread2;
    ret = HcclDedicatedThreadAcquire(comm2, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 3, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread2, 0ULL);
    EXPECT_NE(thread1, thread2);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_UseTypeAicpuUnfoldGe_NotInMap_Return_HCCL_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    ThreadHandle thread;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_UseTypeAicpuUnfoldGe_InMap_Return_SameThread)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    // 先通过 AICPU_UNFOLD 创建专用线程，再将句柄复制到 AICPU_UNFOLD_GE 槽位，模拟已存在场景
    ThreadHandle unfoldThread;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 3, &unfoldThread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(unfoldThread, 0ULL);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto* threadMgr = hcclComm->GetCollComm()->GetCommEngineResMgr()->threadMgr_.get();
    threadMgr->dedicatedThreadMap_[HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE] = unfoldThread;

    ThreadHandle thread;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, unfoldThread);
}

TEST_F(
    TestHcclThread,
    Ut_HcclDedicatedThreadAcquire_When_UseTypeAicpuUnfoldGe_RepeatedAcquire_Return_SameThreadAndNotifyPlus)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    // 先通过 AICPU_UNFOLD 创建专用线程（notifyNum=2），再将句柄复制到 AICPU_UNFOLD_GE 槽位
    ThreadHandle unfoldThread;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &unfoldThread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(unfoldThread, 0ULL);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto* threadMgr = hcclComm->GetCollComm()->GetCommEngineResMgr()->threadMgr_.get();
    threadMgr->dedicatedThreadMap_[HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE] = unfoldThread;

    // 第一次获取 GE 线程，notifyNum=3，需补充1个notify
    ThreadHandle thread1;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 3, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread1, unfoldThread);
    Thread* threadPtr1 = reinterpret_cast<Thread*>(thread1);
    EXPECT_EQ(threadPtr1->GetNotifyNum(), 3U);

    // 第二次获取 GE 线程，notifyNum=5，需补充2个notify
    ThreadHandle thread2;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 5, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread2, thread1);
    Thread* threadPtr2 = reinterpret_cast<Thread*>(thread2);
    EXPECT_EQ(threadPtr2->GetNotifyNum(), 5U);
}

// ============ RFC 0002: L0 HcommThread* 新接口 UT ============

// HcommThreadAlloc per-thread notifyNum 各异（数组版）
TEST_F(TestHcclThread, Ut_HcommThreadAlloc_When_PerThreadNotifyNumDistinct_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[2];
    uint32_t notifyNums[2] = {2, 4}; // 各异
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, notifyNums, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Thread* t0 = reinterpret_cast<Thread*>(thread[0]);
    Thread* t1 = reinterpret_cast<Thread*>(thread[1]);
    EXPECT_EQ(t0->GetNotifyNum(), 2U);
    EXPECT_EQ(t1->GetNotifyNum(), 4U);
    ret = HcommThreadFree(thread, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommThreadAlloc per-thread 校验：单元素超出旧上限 64 后应成功（规格放开，由底层资源决定）
TEST_F(TestHcclThread, Ut_HcommThreadAlloc_When_PerThreadNotifyExceedsOldLimit_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[2];
    uint32_t notifyNums[2] = {2, hccl::HCOMM_NOTIFY_MAX_NUM + 1}; // 第二个超出旧上限
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, notifyNums, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommThreadFree(thread, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommThreadAllocWithStream + HcommThreadFree 无泄漏（AICPU 不支持 stream 入参）
TEST_F(TestHcclThread, Ut_HcommThreadAllocWithStream_When_AllocThenFree_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 进程级 g_ThreadMap 已记，Free 后应可 erase
    ret = HcommThreadFree(&thread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete stream;
}

// HcommThreadSupplementNotify：单线程补充 notify
TEST_F(TestHcclThread, Ut_HcommThreadSupplementNotify_When_Single_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 2;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ThreadHandle handle = thread;
    uint32_t addNotifyNum = 3; // 补 3（增量）
    ret = HcommThreadSupplementNotify(&handle, 1, &addNotifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Thread* t = reinterpret_cast<Thread*>(handle);
    EXPECT_EQ(t->GetNotifyNum(), 5U);
    ret = HcommThreadFree(&thread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommThreadSupplementNotify：空指针
TEST_F(TestHcclThread, Ut_HcommThreadSupplementNotify_When_NullHandle_Expect_E_PTR)
{
    uint32_t addNotifyNum = 3;
    HcommResult ret = HcommThreadSupplementNotify(nullptr, 1, &addNotifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// HcommThreadGetNotifyNum：查询
TEST_F(TestHcclThread, Ut_HcommThreadGetNotifyNum_When_Normal_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 3;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t out = 0;
    ret = HcommThreadGetNotifyNum(thread, &out);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(out, 3U);
    ret = HcommThreadFree(&thread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommThreadExportToCommEngine：CPU->AICPU 导出
TEST_F(TestHcclThread, Ut_HcommThreadExportToCommEngine_When_CpuToAicpu_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForComm));
    ThreadHandle hostThread;
    uint32_t notifyNum = 2;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &hostThread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ThreadHandle devThread = 0;
    std::string commId = "";
    ret = HcommThreadExportToCommEngine(&hostThread, commId.c_str(), 1, COMM_ENGINE_AICPU_TS, &devThread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(devThread, 0ULL);
    ret = HcommThreadFree(&hostThread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommThreadExportToCommEngine：空指针
TEST_F(TestHcclThread, Ut_HcommThreadExportToCommEngine_When_NullHandle_Expect_E_PTR)
{
    ThreadHandle devThread = 0;
    std::string commId = "";
    HcommResult ret = HcommThreadExportToCommEngine(nullptr, commId.c_str(), 1, COMM_ENGINE_AICPU_TS, &devThread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

static HcclResult InitThreadEntityInTestKernel(
    rtStream_t, void* args, u32 size, aclrtBinHandle, const std::string&, bool, u16, void*, u32)
{
    if (args == nullptr || size != sizeof(uint64_t)) {
        return HCCL_E_PARA;
    }
    const uint64_t context = *static_cast<const uint64_t*>(args);
    auto* param = reinterpret_cast<ThreadMgrAicpuParam*>(context);
    EXPECT_NE(param->deviceHandle, nullptr);
    if (param->deviceHandle == nullptr) {
        return HCCL_E_PTR;
    }
    auto** entities = static_cast<HcommThreadEntity**>(param->deviceHandle);
    for (u32 i = 0; i < param->threadNum; ++i) {
        entities[i]->compatThreadHandle = static_cast<ThreadHandle>(0x4000 + i * 0x100);
    }
    return HCCL_SUCCESS;
}

TEST_F(TestHcclThread, StoreAicpuThreadHandlesReturnsEntityInsteadOfRawDeviceHandle)
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunch).stubs().will(invoke(InitThreadEntityInTestKernel));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<std::shared_ptr<Thread>> hostThreads{
        std::make_shared<SpyThreadForFree>(), std::make_shared<SpyThreadForFree>()};
    ASSERT_EQ(SaveThreads(hostThreads), HCCL_SUCCESS);
    ThreadHandle handles[2]{};
    ASSERT_EQ(
        StoreThreadHandles(hostThreads, handles, COMM_ENGINE_AICPU_TS, reinterpret_cast<aclrtBinHandle>(1)),
        HCCL_SUCCESS);
    for (size_t i = 0; i < hostThreads.size(); ++i) {
        EXPECT_NE(handles[i], 0x4000 + i * 0x100);
        EXPECT_EQ(hostThreads[i]->FindThreadHandleByCommEngine(COMM_ENGINE_AICPU_TS), handles[i]);
        ThreadHandle rawHandle = handles[i];
        ASSERT_EQ(UnwrapThreadHandle(rawHandle), HCCL_SUCCESS);
        EXPECT_EQ(rawHandle, 0x4000 + i * 0x100);
        std::shared_ptr<Thread> resolved;
        ASSERT_EQ(LookupThreadByHandle(handles[i], resolved), HCCL_SUCCESS);
        EXPECT_EQ(resolved, hostThreads[i]);
    }
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(FreeThreads(handles, 2, nullptr), HCCL_SUCCESS);
}

TEST_F(TestHcclThread, A3ExportReusesEntityHandleAndResolvesHost)
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910_93)).will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunch).expects(once()).will(invoke(InitThreadEntityInTestKernel));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    ThreadHandle hostHandle{};
    uint32_t notifyNum = 1;
    ASSERT_EQ(HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &hostHandle), HCCL_SUCCESS);
    ThreadHandle entityHandle{};
    ASSERT_EQ(
        HcommThreadExportToCommEngine(&hostHandle, "entity_ut", 1, COMM_ENGINE_AICPU_TS, &entityHandle), HCCL_SUCCESS);
    ThreadHandle rawHandle = entityHandle;
    ASSERT_EQ(UnwrapThreadHandle(rawHandle), HCCL_SUCCESS);
    EXPECT_EQ(rawHandle, 0x4000);
    EXPECT_NE(entityHandle, rawHandle);
    ThreadHandle reused{};
    ASSERT_EQ(HcommThreadExportToCommEngine(&hostHandle, "entity_ut", 1, COMM_ENGINE_AICPU_TS, &reused), HCCL_SUCCESS);
    EXPECT_EQ(reused, entityHandle);
    ThreadHandle returnedHost{};
    ASSERT_EQ(
        HcommThreadExportToCommEngine(&entityHandle, "entity_ut", 1, COMM_ENGINE_CPU_TS, &returnedHost), HCCL_SUCCESS);
    EXPECT_EQ(returnedHost, hostHandle);
    EXPECT_EQ(HcommThreadFree(&entityHandle, 1), HCCL_SUCCESS);
}

TEST_F(TestHcclThread, GeFakeExportReturnsRawHandleAndPreservesNormalEntity)
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunch).expects(once()).will(invoke(InitThreadEntityInTestKernel));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchDestroy).expects(never());
    ThreadHandle hostHandles[2]{};
    uint32_t notifyNum = 1;
    ASSERT_EQ(HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &hostHandles[0]), HCCL_SUCCESS);
    auto* normalThread = reinterpret_cast<Thread*>(hostHandles[0]);
    void* notifys[] = {normalThread->GetNotify(0)};
    ASSERT_EQ(HcommThreadAcquireByNotify(normalThread->GetStream()->ptr(), notifys, 1, &hostHandles[1]), HCCL_SUCCESS);
    auto* fakeThread = reinterpret_cast<Thread*>(hostHandles[1]);
    ASSERT_TRUE(fakeThread->IsFakeDeviceRes());

    ThreadHandle exported[2]{};
    ASSERT_EQ(HcommThreadExportToCommEngine(hostHandles, "ge_ctx_ut", 2, COMM_ENGINE_AICPU_TS, exported), HCCL_SUCCESS);
    auto* normalResources = normalThread->GetThreadEntityResources(COMM_ENGINE_AICPU_TS);
    ASSERT_NE(normalResources, nullptr);
    EXPECT_EQ(exported[0], normalResources->GetEntityHandle());
    EXPECT_EQ(reinterpret_cast<HcommThreadEntity*>(exported[0])->compatThreadHandle, 0x4000U);
    EXPECT_EQ(exported[1], 0x4100U); // 混合批次按原始下标读取兼容句柄。
    EXPECT_EQ(fakeThread->GetThreadEntityResources(COMM_ENGINE_AICPU_TS), nullptr);
    EXPECT_EQ(fakeThread->FindThreadHandleByCommEngine(COMM_ENGINE_AICPU_TS), exported[1]);
    EXPECT_EQ(fakeThread->GetNotify(0), notifys[0]); // 仅回收 Entity 描述，不销毁借用的 notify。

    ThreadHandle reused[2]{};
    ASSERT_EQ(HcommThreadExportToCommEngine(hostHandles, "ge_ctx_ut", 2, COMM_ENGINE_AICPU_TS, reused), HCCL_SUCCESS);
    EXPECT_EQ(reused[0], exported[0]);
    EXPECT_EQ(reused[1], exported[1]);
    EXPECT_EQ(fakeThread->GetThreadEntityResources(COMM_ENGINE_AICPU_TS), nullptr);
    ThreadHandle returnedHost{};
    ASSERT_EQ(
        HcommThreadExportToCommEngine(&exported[1], "ge_ctx_ut", 1, COMM_ENGINE_CPU_TS, &returnedHost), HCCL_SUCCESS);
    EXPECT_EQ(returnedHost, hostHandles[1]);
    EXPECT_EQ(HcommThreadFree(&hostHandles[1], 1), HCCL_SUCCESS);
    EXPECT_EQ(HcommThreadFree(&hostHandles[0], 1), HCCL_SUCCESS);
}

TEST_F(TestHcclThread, A3GeFakeExportReturnsRawHandle)
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910_93)).will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunch).expects(once()).will(invoke(InitThreadEntityInTestKernel));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<std::shared_ptr<Thread>> threads{std::make_shared<SpyThreadForFree>()};
    threads[0]->SetCommEngine(COMM_ENGINE_CPU_TS);
    threads[0]->SetFakeDeviceRes();
    ASSERT_EQ(SaveThreads(threads), HCCL_SUCCESS);
    ThreadHandle hostHandle = reinterpret_cast<ThreadHandle>(threads[0].get());
    ThreadHandle exported{};
    ASSERT_EQ(
        HcommThreadExportToCommEngine(&hostHandle, "ge_ctx_ut", 1, COMM_ENGINE_AICPU_TS, &exported), HCCL_SUCCESS);
    EXPECT_EQ(exported, 0x4000U);
    EXPECT_EQ(threads[0]->GetThreadEntityResources(COMM_ENGINE_AICPU_TS), nullptr);
    EXPECT_EQ(HcommThreadFree(&hostHandle, 1), HCCL_SUCCESS);
}

TEST_F(TestHcclThread, GeFakeExportKeepsEntityWhenKernelSynchronizationFails)
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunch).expects(once()).will(invoke(InitThreadEntityInTestKernel));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_E_TIMEOUT));
    std::vector<std::shared_ptr<Thread>> threads{std::make_shared<SpyThreadForFree>()};
    threads[0]->SetCommEngine(COMM_ENGINE_CPU_TS);
    threads[0]->SetFakeDeviceRes();
    ASSERT_EQ(SaveThreads(threads), HCCL_SUCCESS);
    ThreadHandle hostHandle = reinterpret_cast<ThreadHandle>(threads[0].get());
    ThreadHandle exported{};
    EXPECT_EQ(
        HcommThreadExportToCommEngine(&hostHandle, "ge_ctx_ut", 1, COMM_ENGINE_AICPU_TS, &exported), HCCL_E_TIMEOUT);
    EXPECT_EQ(exported, 0U);
    EXPECT_EQ(threads[0]->FindThreadHandleByCommEngine(COMM_ENGINE_AICPU_TS), 0U);
    EXPECT_NE(threads[0]->GetThreadEntityResources(COMM_ENGINE_AICPU_TS), nullptr);
    // UT kernel 同步写完，无在途任务；仅在完成上述保留断言后清理 Host 对象。
    EXPECT_EQ(HcommThreadFree(&hostHandle, 1), HCCL_SUCCESS);
}

TEST_F(TestHcclThread, GeFakeExportKeepsEntityWhenCompatHandleReadFails)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    SpyThreadForFree thread;
    thread.SetFakeDeviceRes();
    ThreadHandle handle{};
    ASSERT_EQ(thread.PreAllocThreadEntity(handle, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    const ThreadHandle entityHandle = handle;
    reinterpret_cast<HcommThreadEntity*>(handle)->compatThreadHandle = 0x4000;
    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_E_RUNTIME));
    EXPECT_EQ(thread.ConvertThreadEntityToRawHandle(handle, COMM_ENGINE_AICPU_TS), HCCL_E_RUNTIME);
    EXPECT_EQ(handle, entityHandle);
    EXPECT_EQ(thread.FindThreadHandleByCommEngine(COMM_ENGINE_AICPU_TS), 0U);
    EXPECT_NE(thread.GetThreadEntityResources(COMM_ENGINE_AICPU_TS), nullptr);
}

TEST_F(TestHcclThread, GeFakeExportRejectsUninitializedAndPublishedEntity)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    SpyThreadForFree thread;
    thread.SetFakeDeviceRes();
    ThreadHandle handle{};
    ASSERT_EQ(thread.PreAllocThreadEntity(handle, COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    const ThreadHandle entityHandle = handle;
    EXPECT_EQ(thread.ConvertThreadEntityToRawHandle(handle, COMM_ENGINE_AICPU_TS), HCCL_E_INTERNAL);
    EXPECT_EQ(handle, entityHandle);
    EXPECT_NE(thread.GetThreadEntityResources(COMM_ENGINE_AICPU_TS), nullptr);

    reinterpret_cast<HcommThreadEntity*>(handle)->compatThreadHandle = 0x4000;
    ASSERT_EQ(thread.AddThreadHandleToMap(COMM_ENGINE_AICPU_TS, handle), HCCL_SUCCESS);
    EXPECT_EQ(thread.ConvertThreadEntityToRawHandle(handle, COMM_ENGINE_AICPU_TS), HCCL_E_PARA);
    EXPECT_EQ(handle, entityHandle);
    EXPECT_NE(thread.GetThreadEntityResources(COMM_ENGINE_AICPU_TS), nullptr);
}

TEST_F(TestHcclThread, A5HostPreallocatesPodAndPreservesDeviceStateOnGrowth)
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    std::shared_ptr<Thread> thread;
    ASSERT_EQ(
        CreateThread(
            COMM_ENGINE_AICPU_TS, THREAD_TYPE_TS, StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::HOST_NOTIFY,
            thread),
        HCCL_SUCCESS);
    ASSERT_EQ(thread->Init(), HCCL_SUCCESS);
    ThreadHandle handles[2]{};
    ASSERT_EQ(thread->PreAllocThreadEntity(handles[0], COMM_ENGINE_AICPU), HCCL_SUCCESS);
    ASSERT_EQ(thread->PreAllocThreadEntity(handles[1], COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    auto* entity = reinterpret_cast<HcommThreadEntity*>(handles[0]);
    auto* other = reinterpret_cast<HcommThreadEntity*>(handles[1]);
    EXPECT_EQ(entity->abiHeader.magicWord, 0x0fcf0f6fU);
    EXPECT_EQ(entity->abiHeader.size, 256U);
    EXPECT_EQ(entity->localNotifyNum, 2U);
    ASSERT_NE(entity->localNotifyAddr, nullptr);
    EXPECT_EQ(entity->localNotifyAddr[0].notifyInfo.ipcRt.addr, 0U);
    ASSERT_NE(entity->sqContextAddr, nullptr);
    EXPECT_EQ(reinterpret_cast<ThreadHandle>(entity->sqContextAddr), handles[0] + 256U);
    EXPECT_NE(entity->sqContextAddr, other->sqContextAddr);
    auto* sq = entity->sqContextAddr;
    EXPECT_EQ(sq->contextInfo.RtsqContext.streamId, 0U);
    EXPECT_EQ(sq->contextInfo.RtsqContext.sqId, 0U);
    EXPECT_EQ(sq->type, SQ_CONTEXT_TYPE_UB_JFS);
    EXPECT_EQ(sq->contextInfo.RtsqContext.sqDepth, 0U);
    EXPECT_EQ(sq->contextInfo.RtsqContext.sqBaseAddr, 0U);
    EXPECT_EQ(sq->contextInfo.RtsqContext.sqHead, 0U);
    EXPECT_EQ(sq->contextInfo.RtsqContext.taskId, 0U);
    EXPECT_EQ(sq->contextInfo.RtsqContext.localSqDepth, 128U);
    ASSERT_NE(sq->contextInfo.RtsqContext.localSqAddr, 0U);
    const auto* cache
        = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(sq->contextInfo.RtsqContext.localSqAddr));
    const uint32_t cacheSize = sq->contextInfo.RtsqContext.localSqDepth * 64U;
    EXPECT_EQ(cache[0], 0U);
    EXPECT_EQ(cache[cacheSize - 1], 0U);
    entity->compatThreadHandle = static_cast<ThreadHandle>(0x1230);
    entity->localNotifyNum = 2;
    entity->localNotifyAddr[0].notifyInfo.ipcRt.notifyId = 17;
    sq->contextInfo.RtsqContext.sqHead = 13;
    sq->contextInfo.RtsqContext.taskId = 7;
    ASSERT_EQ(thread->SupplementNotify(1), HCCL_SUCCESS);
    EXPECT_EQ(entity->localNotifyNum, 2U); // 实际 Notify 的补充不再隐式修改 Entity。
    EXPECT_EQ(other->localNotifyNum, 2U);
    for (CommEngine engine : {COMM_ENGINE_AICPU, COMM_ENGINE_AICPU_TS}) {
        auto* resources = thread->GetThreadEntityResources(engine);
        ASSERT_NE(resources, nullptr);
        ASSERT_EQ(resources->UpdateNotifyBuffer(thread->GetNotifyNum()), HCCL_SUCCESS);
    }
    EXPECT_EQ(entity->localNotifyNum, 3U);
    EXPECT_EQ(other->localNotifyNum, 3U);
    EXPECT_EQ(entity->localNotifyAddr[0].notifyInfo.ipcRt.notifyId, 17);
    EXPECT_EQ(entity->compatThreadHandle, static_cast<ThreadHandle>(0x1230));
    EXPECT_EQ(entity->sqContextAddr, sq);
    EXPECT_EQ(sq->contextInfo.RtsqContext.sqHead, 13U);
    EXPECT_EQ(sq->contextInfo.RtsqContext.taskId, 7U);
    ThreadHandle reused{};
    ASSERT_EQ(thread->PreAllocThreadEntity(reused, COMM_ENGINE_AICPU), HCCL_SUCCESS);
    EXPECT_EQ(reused, handles[0]);
}

static std::vector<ThreadHandle> g_supplementedExports;
static HcclResult CaptureSupplementExports(
    std::vector<std::shared_ptr<hccl::Thread>>& threads, const std::string& commId,
    std::unique_ptr<ThreadHandle[]>& handles, aclrtBinHandle binHandle)
{
    for (size_t i = 0; i < threads.size(); ++i) {
        auto* entity = reinterpret_cast<HcommThreadEntity*>(handles[i]);
        EXPECT_EQ(entity->localNotifyNum, threads[i]->GetNotifyNum());
        EXPECT_NE(entity->localNotifyAddr, nullptr);
        g_supplementedExports.push_back(handles[i]);
    }
    return HCCL_SUCCESS;
}

TEST_F(TestHcclThread, AicpuSupplementUpdatesEntityBeforeExistingDeviceLaunch)
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchForBase).stubs().will(invoke(StubThreadKernelLaunchForBase));
    MOCKER_CPP(&AicpuThreadLaunchMgr::ThreadKernelLaunchDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadLaunchMgr::SupplementNotifyKernelLaunch)
        .expects(once())
        .will(invoke(CaptureSupplementExports));
    g_supplementedExports.clear();
    ThreadHandle handle{};
    uint32_t notifyNum = 1;
    ASSERT_EQ(HcommThreadAlloc(COMM_ENGINE_AICPU, 1, &notifyNum, &handle), HCCL_SUCCESS);
    const ThreadHandle originalHandle = handle;
    uint32_t increment = 1;
    ASSERT_EQ(HcommThreadSupplementNotify(&handle, 1, &increment), HCCL_SUCCESS);
    ASSERT_EQ(g_supplementedExports.size(), 1U);
    EXPECT_EQ(g_supplementedExports[0], originalHandle);
    EXPECT_EQ(reinterpret_cast<HcommThreadEntity*>(handle)->localNotifyNum, 2U);
    EXPECT_EQ(HcommThreadFree(&handle, 1), HCCL_SUCCESS);
}
