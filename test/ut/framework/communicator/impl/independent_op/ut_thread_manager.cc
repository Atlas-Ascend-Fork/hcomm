/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "hccl/hccl_res.h"
#include "hcomm_c_adpt.h"
#include "hcomm_thread_c_adpt.h"
#include "hcclCommOp.h"
#include "../../hccl_api_base_test.h"
#include "hccl_tbe_task.h"
#include "thread_manager.h"
#include "launch_aicpu.h"
#include "aicpu_launch_manager.h"
#include "adapter_rts_common.h"
#include "adapter_rts.h"
#include "aicpu_ts_thread.h"

using namespace hccl;

static HcclResult StubThreadKernelLaunchForCommDevice(
    std::vector<std::shared_ptr<hccl::Thread>>& newThreads, const std::string& commId,
    std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle)
{
    for (size_t i = 0; i < newThreads.size(); ++i) {
        aicpuHandle[i] = static_cast<ThreadHandle>(0x5000 + i);
    }
    return HCCL_SUCCESS;
}
void MockGetRunSideIsDevice();
void MockThreadKernelLaunchForComm();

// 流ID查询桩：须定义于fixture类之前（SetUp引用）。UT用假流指针（如0x1234），
// 真实hrtGetStreamId→aclrtStreamGetId会解引用句柄读magic校验→段错误；
// fixture级mock保证全部用例安全，返回值仅用于审计日志
static HcclResult StubHrtGetStreamId(void* stream, s32& streamId)
{
    (void)stream;
    streamId = 1;
    return HCCL_SUCCESS;
}

namespace {
// EnsureAicpuCommInit spy：记录 init/alloc 调用次数与顺序（invoke 桩无法捕获变量，用文件级静态对象共享）
struct AicpuCommInitSpy {
    uint32_t seq = 0;      // 全局调用序号
    uint32_t initSeq = 0;  // kernelLaunchAicpuCommInit 首次调用序号
    uint32_t allocSeq = 0; // HcommThreadAllocWithCommConfig 首次调用序号
    uint32_t initCnt = 0;
    uint32_t allocCnt = 0;
    HcclResult initRet = HCCL_SUCCESS;
    bool commState = false;
};
AicpuCommInitSpy g_aicpuSpy;
} // namespace

// L0 线程分配桩：填充伪句柄并记录调用顺序
static HcommResult StubThreadAllocWithCommConfig(
    CommEngine, const char*, uint32_t threadNum, ThreadType, const ThreadConfig*, ThreadHandle* threads)
{
    g_aicpuSpy.allocCnt++;
    if (g_aicpuSpy.allocSeq == 0) {
        g_aicpuSpy.allocSeq = ++g_aicpuSpy.seq;
    }
    for (uint32_t i = 0; i < threadNum; ++i) {
        threads[i] = static_cast<ThreadHandle>(0x7000 + i);
    }
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// L0 线程资源查询桩：返回伪 stream 指针
static HcommResult StubThreadResGetInfo(ThreadHandle, ThreadResType, uint32_t, void** info)
{
    *info = reinterpret_cast<void*>(0x7100);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// 重置 spy 并 mock AICPU host 线程分配链路的 L0 边界，使 UT 不依赖真实线程资源
static void MockAicpuCommInitEnv()
{
    g_aicpuSpy = AicpuCommInitSpy();
    MOCKER(HcommThreadAllocWithCommConfig).stubs().will(invoke(StubThreadAllocWithCommConfig));
    MOCKER(HcommThreadResGetInfo).stubs().will(invoke(StubThreadResGetInfo));
    MOCKER(hrtStreamGetSqid).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadSupplementNotify).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(HcommThreadFree).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
}

class ThreadManagerTest : public BaseInit {
public:
    void SetUp() override
    {
        std::cout << "ThreadManagerTest SetUp" << std::endl;
        BaseInit::SetUp();
        MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
        // GE链路审计日志含流ID真实查询，UT假流指针须mock（见StubHrtGetStreamId注释）
        MOCKER(hrtGetStreamId).stubs().will(invoke(StubHrtGetStreamId));
        ManagerCallbacks callbacks;
        callbacks.getAicpuCommState = []() {
            return true;
        };
        callbacks.setAicpuCommState = [](bool) {};
        callbacks.kernelLaunchAicpuCommInit = []() {
            return HCCL_SUCCESS;
        };
        callbacks.reportProfilingKernel = [](uint64_t, std::string) {
            return HCCL_SUCCESS;
        };
        // binHandle 只透传给 DpuThreadService 的 taskexp 下发器，本组用例不触发下发，传空即可
        threadManager = std::make_unique<ThreadMgr>(4, 8, "test", nullptr, callbacks);
    }
    void TearDown() override
    {
        std::cout << "ThreadManagerTest TearDown" << std::endl;
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

    // 申请 N 个 CPU 线程（HcclThreadAcquireV2），notifyNum 一致
    void AcquireCpuThreads(uint32_t threadNum, uint16_t notifyNum, ThreadHandle* out)
    {
        MockGetRunSideIsDevice();
        std::vector<ThreadConfig> config(threadNum);
        EXPECT_EQ(ThreadConfigInit(config.data(), threadNum), HCCL_SUCCESS);
        for (uint32_t i = 0; i < threadNum; ++i) {
            config[i].notifyNumPerThread = notifyNum;
        }
        std::vector<uint32_t> threadId;
        EXPECT_EQ(
            threadManager->HcclThreadAcquireV2(
                CommEngine::COMM_ENGINE_CPU, threadNum, ThreadType::THREAD_TYPE_TS, config.data(), out, threadId),
            HCCL_SUCCESS);
    }

    // AICPU 公共域 spy 回调组：init 状态与次数可控可观测
    ManagerCallbacks MakeAicpuSpyCallbacks()
    {
        ManagerCallbacks callbacks;
        callbacks.getAicpuCommState = []() {
            return g_aicpuSpy.commState;
        };
        callbacks.setAicpuCommState = [](bool state) {
            g_aicpuSpy.commState = state;
        };
        callbacks.kernelLaunchAicpuCommInit = []() {
            g_aicpuSpy.initCnt++;
            if (g_aicpuSpy.initSeq == 0) {
                g_aicpuSpy.initSeq = ++g_aicpuSpy.seq;
            }
            return g_aicpuSpy.initRet;
        };
        callbacks.reportProfilingKernel = [](uint64_t, std::string) {
            return HCCL_SUCCESS;
        };
        return callbacks;
    }

private:
    std::unique_ptr<ThreadMgr> threadManager;
    uint32_t threadNum = 1;
    ThreadHandle threads[1] = {0};
    ThreadHandle exportedThreads[1] = {0};
};

void MockGetRunSideIsDevice()
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
}

void MockThreadKernelLaunchForComm()
{
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(HCCL_SUCCESS));
}

TEST_F(ThreadManagerTest, Ut_ThreadExportToCommEngineAicpu_When_InvalidThreadHandle_Expect_HCCL_E_NOT_FOUND)
{
    CommEngine dstCommEngine = COMM_ENGINE_AICPU_TS;

    HcclResult ret = threadManager->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(ThreadManagerTest, Ut_ThreadExportToCommEngineAicpu_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MockGetRunSideIsDevice();
    MockThreadKernelLaunchForComm();

    CommEngine dstCommEngine = COMM_ENGINE_AICPU_TS;

    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    if (ret == HCCL_SUCCESS) {
        ret = threadManager->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_NoThreads_Expect_Success)
{
    EXPECT_EQ(threadManager->ResetThreadLocalNotifies(), HCCL_SUCCESS);
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_ResetNotifiesFailed_Expect_ReturnFailed)
{
    MockGetRunSideIsDevice();
    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    MOCKER(HcommThreadResetNotifies).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    ret = threadManager->ResetThreadLocalNotifies();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_OrderLaunchThreadRegistered_Expect_Success)
{
    MockGetRunSideIsDevice();
    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    ret = threadManager->RegisterOrderLaunchThread(threads[0]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(threadManager->ResetThreadLocalNotifies(), HCCL_SUCCESS);
}

/* ======================== HcclDedicatedThreadAcquire DEVICE ======================== */

static void MockAicpuThreadEnv()
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceTypeInvalid_Expect_HCCL_E_PARA)
{
    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_INVALID, 1, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceThreadNullptr_Expect_HCCL_E_PTR)
{
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceCreateSuccess_Expect_NonZeroThread)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForCommDevice));

    ThreadHandle thread = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceRepeatedAcquire_Expect_SameThread)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForCommDevice));

    ThreadHandle thread1 = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread1, static_cast<ThreadHandle>(0));

    ThreadHandle thread2 = 0;
    ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread1, thread2);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceKernelLaunchFail_Expect_Error)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(HCCL_E_INTERNAL));

    ThreadHandle thread = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ============ RFC 0002: L1 ThreadMgr 归一 UT ============

// HcclThreadAcquireV2：CPU 引擎批量分配 + 返回 sqId
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_CpuBatch_Expect_Success)
{
    MockGetRunSideIsDevice();
    constexpr uint32_t kNum = 2;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    for (uint32_t i = 0; i < kNum; ++i) {
        config[i].notifyNumPerThread = static_cast<uint16_t>(2 + i);
    }
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = threadManager->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(threadId.size(), kNum);
    // 释放（L1 析构不负责，显式 Free）
    HcommThreadFree(out, kNum);
}

// HcclThreadAcquireV2：复用池命中 + notify 补充
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_ReusePoolHit_Expect_SupplementNotify)
{
    MockGetRunSideIsDevice();
    MockThreadKernelLaunchForComm();
    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 2;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = threadManager->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ThreadHandle first = out[0];

    // 第二次：notifyNum 提升到 4，触发 SupplementNotify
    config[0].notifyNumPerThread = 4;
    std::vector<uint32_t> threadId2;
    ret = threadManager->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(out[0], first); // 复用同句柄
    HcommThreadFree(out, kNum);
}

// HcclThreadAcquire：批量分配 + 返回 sqId
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquire_When_NonV2Batch_Expect_Success)
{
    MockGetRunSideIsDevice();
    constexpr uint32_t kNum = 2;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    for (uint32_t i = 0; i < kNum; ++i) {
        config[i].notifyNumPerThread = 2;
    }
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = threadManager->HcclThreadAcquire(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(threadId.size(), kNum);
    HcommThreadFree(out, kNum);
}

// HcclGetNotifyNumInThread：L1 反查
TEST_F(ThreadManagerTest, Ut_HcclGetNotifyNumInThread_When_Normal_Expect_Success)
{
    MockGetRunSideIsDevice();
    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 3;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = threadManager->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_CPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    uint32_t num = 0;
    ret = threadManager->HcclGetNotifyNumInThread(out[0], &num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(num, 3U);
    HcommThreadFree(out, kNum);
}

TEST_F(ThreadManagerTest, Ut_HcclThreadResGetInfo_When_Stream_Expect_Success)
{
    ThreadHandle handle;
    AcquireCpuThreads(1, 1, &handle);
    void* info = nullptr;
    HcclResult ret = threadManager->HcclThreadResGetInfo(
        handle, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &info);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(info, nullptr);
    HcommThreadFree(&handle, 1);
}

// 析构释放：ThreadMgr 析构不泄漏（L1 cache 清空，L0 g_ThreadMap 由 HcommThreadFree 管）
TEST_F(ThreadManagerTest, Ut_ThreadMgrDestructor_When_NoLeak_Expect_Success)
{
    ThreadHandle handle;
    AcquireCpuThreads(1, 1, &handle);
    // 显式 Free 后析构 cache 已空
    EXPECT_EQ(HcommThreadFree(&handle, 1), HCCL_SUCCESS);
}

/* ======================== EnsureAicpuCommInit（AICPU 首扩容场景） ======================== */

// AICPU 首次 acquire（existNum=0）：comm-init 恰好执行一次，且发生在线程分配之前
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_AicpuFirstAcquire_Expect_CommInitOnceBeforeAlloc)
{
    MockGetRunSideIsDevice();
    MockAicpuCommInitEnv();
    std::unique_ptr<ThreadMgr> mgr = std::make_unique<ThreadMgr>(4, 8, "test", nullptr, MakeAicpuSpyCallbacks());

    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 2;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = mgr->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_AICPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_aicpuSpy.initCnt, 1U);
    EXPECT_EQ(g_aicpuSpy.allocCnt, 1U);
    ASSERT_NE(g_aicpuSpy.initSeq, 0U);
    ASSERT_NE(g_aicpuSpy.allocSeq, 0U);
    EXPECT_LT(g_aicpuSpy.initSeq, g_aicpuSpy.allocSeq); // comm-init 先于线程分配
}

// AICPU 重复 acquire：comm-init 防重生效，不重复执行
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_AicpuRepeatedAcquire_Expect_CommInitNotRepeated)
{
    MockGetRunSideIsDevice();
    MockAicpuCommInitEnv();
    std::unique_ptr<ThreadMgr> mgr = std::make_unique<ThreadMgr>(4, 8, "test", nullptr, MakeAicpuSpyCallbacks());

    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 2;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = mgr->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_AICPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(g_aicpuSpy.initCnt, 1U);

    ret = mgr->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_AICPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_aicpuSpy.initCnt, 1U);  // 防重生效
    EXPECT_EQ(g_aicpuSpy.allocCnt, 1U); // 复用池命中，不重复分配
}

// comm-init 失败：acquire 返回错误并早退，不进行线程分配，状态未置位
TEST_F(ThreadManagerTest, Ut_HcclThreadAcquireV2_When_CommInitFailed_Expect_AcquireFailedBeforeAlloc)
{
    MockGetRunSideIsDevice();
    MockAicpuCommInitEnv();
    g_aicpuSpy.initRet = HCCL_E_INTERNAL;
    std::unique_ptr<ThreadMgr> mgr = std::make_unique<ThreadMgr>(4, 8, "test", nullptr, MakeAicpuSpyCallbacks());

    constexpr uint32_t kNum = 1;
    ThreadConfig config[kNum];
    ASSERT_EQ(ThreadConfigInit(config, kNum), HCCL_SUCCESS);
    config[0].notifyNumPerThread = 2;
    ThreadHandle out[kNum] = {0};
    std::vector<uint32_t> threadId;
    HcclResult ret = mgr->HcclThreadAcquireV2(
        CommEngine::COMM_ENGINE_AICPU, kNum, ThreadType::THREAD_TYPE_TS, config, out, threadId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(g_aicpuSpy.initCnt, 1U);
    EXPECT_EQ(g_aicpuSpy.allocCnt, 0U); // 失败早退，未走到线程分配
    EXPECT_FALSE(g_aicpuSpy.commState); // 状态未置位，下次可重试
}
/* ============ GE 展开链路（SetAttachedStream / HcclGeUnfoldThreadAcquire / FreeDedicatedThreads） ============ */

static ThreadHandle g_geAllocHandle = 0x3000; // 模拟 HcommThreadAllocWithStream 产出的线程句柄
static uint32_t g_geAllocCallCount = 0;
static uint32_t g_geSupplementNum = 0;
static uint32_t g_geSupplementCallCount = 0;
static std::vector<ThreadHandle> g_freeCapturedHandles;

static HcommResult StubGeAllocWithStream(CommEngine engine, void* stream, uint32_t notifyNum, ThreadHandle* handle)
{
    (void)engine;
    (void)stream;
    (void)notifyNum;
    *handle = g_geAllocHandle;
    g_geAllocCallCount++;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

static HcommResult
StubGeSupplementNotify(const ThreadHandle* handles, uint32_t threadNum, const uint32_t* supplementNotifyNums)
{
    (void)handles;
    (void)threadNum;
    g_geSupplementNum = *supplementNotifyNums;
    g_geSupplementCallCount++;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// GE展开链路配套mock：HcclGeUnfoldThreadAcquire获取线程后调HcommThreadRegisterDfx，
// 其实现【无句柄表查找】直接ReinterpretAs<Thread*>解引用——mock分配出的假句柄（0x3000）会被
// 裸写成员导致段错误，须与HcommThreadAllocWithStream的mock配套使用
static void MockGeUnfoldDfx() { MOCKER(HcommThreadRegisterDfx).stubs().will(returnValue(0)); }

static HcommResult StubHcommThreadFreeCapture(const ThreadHandle* threads, uint32_t threadNum)
{
    for (uint32_t i = 0; i < threadNum; i++) {
        g_freeCapturedHandles.push_back(threads[i]);
    }
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// SetAttachedStream：空流拒绝
TEST_F(ThreadManagerTest, Ut_SetAttachedStream_When_NullStream_Expect_HCCL_E_PTR)
{
    EXPECT_EQ(threadManager->SetAttachedStream(nullptr), HCCL_E_PTR);
}

// SetAttachedStream：正常流存储（后续GE获取可用的前提）
TEST_F(ThreadManagerTest, Ut_SetAttachedStream_When_Normal_Expect_Success)
{
    EXPECT_EQ(threadManager->SetAttachedStream(reinterpret_cast<void*>(0x1234)), HCCL_SUCCESS);
}

// GE展开获取：未注入附属流 → 降级返回 thread=0
TEST_F(ThreadManagerTest, Ut_GeUnfoldAcquire_When_StreamNotSet_Expect_ZeroThread)
{
    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 2, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, static_cast<ThreadHandle>(0)); // 未SetAttachedStream：告警降级，不报错
}

// GE展开获取：首次创建经 HcommThreadAllocWithStream 并登记 mainThread_
TEST_F(ThreadManagerTest, Ut_GeUnfoldAcquire_When_FirstCreate_Expect_AllocAndRegister)
{
    ASSERT_EQ(threadManager->SetAttachedStream(reinterpret_cast<void*>(0x1234)), HCCL_SUCCESS);
    g_geAllocCallCount = 0;
    MOCKER(HcommThreadAllocWithStream).stubs().will(invoke(StubGeAllocWithStream));
    MockGeUnfoldDfx();

    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 2, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, g_geAllocHandle);
    EXPECT_EQ(g_geAllocCallCount, 1U); // 首次创建走分配
}

// GE展开获取：同流二次调用复用（不发生二次分配）
TEST_F(ThreadManagerTest, Ut_GeUnfoldAcquire_When_SameStreamReacquire_Expect_ReuseWithoutAlloc)
{
    ASSERT_EQ(threadManager->SetAttachedStream(reinterpret_cast<void*>(0x1234)), HCCL_SUCCESS);
    g_geAllocCallCount = 0;
    MOCKER(HcommThreadAllocWithStream).stubs().will(invoke(StubGeAllocWithStream));
    MockGeUnfoldDfx();

    ThreadHandle thread1 = 0;
    ASSERT_EQ(
        threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 2, &thread1), HCCL_SUCCESS);
    ThreadHandle thread2 = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 2, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread2, thread1);       // 同流复用：同句柄
    EXPECT_EQ(g_geAllocCallCount, 1U); // 未发生第二次分配
}

// GE展开获取：notifyNumPerThread 增大 → 复用分支按记账差值补充
TEST_F(ThreadManagerTest, Ut_GeUnfoldAcquire_When_NotifyNumIncreased_Expect_SupplementIncrement)
{
    ASSERT_EQ(threadManager->SetAttachedStream(reinterpret_cast<void*>(0x1234)), HCCL_SUCCESS);
    MOCKER(HcommThreadAllocWithStream).stubs().will(invoke(StubGeAllocWithStream));
    MockGeUnfoldDfx();
    g_geSupplementCallCount = 0;
    g_geSupplementNum = 0;
    MOCKER(HcommThreadSupplementNotify).stubs().will(invoke(StubGeSupplementNotify));

    ThreadHandle thread1 = 0;
    ASSERT_EQ(
        threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 2, &thread1), HCCL_SUCCESS);

    // notifyNum 2→5：复用分支按 ThreadMeta 记账差值补充（5-2=3）
    ThreadHandle thread2 = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 5, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread2, thread1);
    EXPECT_EQ(g_geSupplementCallCount, 1U);
    EXPECT_EQ(g_geSupplementNum, 3U); // 增量 = 目标值 - 已登记值
}

// 析构释放：dedicatedThreadMap_ 全类型句柄经 FreeDedicatedThreads→HcommThreadFree 释放
TEST_F(ThreadManagerTest, Ut_ThreadMgrDestructor_When_DedicatedThreadAcquired_Expect_FreedByHcommThreadFree)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForCommDevice));
    ThreadHandle thread = 0;
    ASSERT_EQ(
        threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread),
        HCCL_SUCCESS);
    ASSERT_NE(thread, static_cast<ThreadHandle>(0));

    g_freeCapturedHandles.clear();
    MOCKER(HcommThreadFree).stubs().will(invoke(StubHcommThreadFreeCapture));
    threadManager.reset(); // 触发~ThreadMgr → FreeDedicatedThreads
    ASSERT_EQ(g_freeCapturedHandles.size(), 1U);
    EXPECT_EQ(g_freeCapturedHandles[0], thread); // 专用线程句柄经HcommThreadFree批量释放
}

/* ============ AICPU_LAUNCH（非GE展开）链路：DFX回调经记录接口下发，首建点注册 ============ */

static ThreadHandle g_unfoldAllocHandle = 0x4000; // 模拟 HcommThreadAlloc 产出的线程句柄
static uint32_t g_unfoldAllocCallCount = 0;
static uint32_t g_dfxRegisterCallCount = 0;
static int32_t g_dfxRegisterRet = 0;

// 非GE展开首建链路 L0 分配桩：填充伪句柄并计数
static HcommResult StubUnfoldAlloc(CommEngine, uint32_t, const uint32_t*, ThreadHandle* threads)
{
    *threads = g_unfoldAllocHandle;
    g_unfoldAllocCallCount++;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// DFX注册桩：计数并透传可控返回值（真实实现ReinterpretAs<Thread*>解引用假句柄会段错误，须mock）
static int32_t StubThreadRegisterDfx(ThreadHandle, std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)>)
{
    g_dfxRegisterCallCount++;
    return g_dfxRegisterRet;
}

// 复用路径 notify 查询桩：返回足量，避免伪句柄经真实 GetNotifyNum/SupplementNotify 解引用
static HcommResult StubGetNotifyNumSufficient(ThreadHandle, uint32_t* notifyNum)
{
    *notifyNum = 100;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

static void MockUnfoldDfxEnv()
{
    g_unfoldAllocCallCount = 0;
    g_dfxRegisterCallCount = 0;
    g_dfxRegisterRet = 0;
    MOCKER(HcommThreadAlloc).stubs().will(invoke(StubUnfoldAlloc));
    MOCKER(HcommThreadRegisterDfx).stubs().will(invoke(StubThreadRegisterDfx));
    MOCKER(HcommThreadGetNotifyNum).stubs().will(invoke(StubGetNotifyNumSufficient));
    MOCKER(HcommThreadFree).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS))); // 析构安全兜底
}

static std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> MakeUnfoldDfxCallback()
{
    return [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
        (void)streamId;
        (void)taskId;
        (void)taskParam;
        (void)handle;
        return HCCL_SUCCESS;
    };
}

// AICPU_LAUNCH首次创建：先记录回调，经HcommThreadAlloc分配并在首建点注册
TEST_F(ThreadManagerTest, Ut_UnfoldAcquire_When_FirstCreateWithDfx_Expect_RegisterDfxOnce)
{
    MockUnfoldDfxEnv();
    ASSERT_EQ(threadManager->HcclDedicatedThreadDfx(MakeUnfoldDfxCallback()), HCCL_SUCCESS);
    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, g_unfoldAllocHandle);
    EXPECT_EQ(g_unfoldAllocCallCount, 1U);
    EXPECT_EQ(g_dfxRegisterCallCount, 1U); // 首次创建注册一次
}

// AICPU_LAUNCH注册失败：按致命处理，返回HCCL_E_INTERNAL
TEST_F(ThreadManagerTest, Ut_UnfoldAcquire_When_DfxRegisterFail_Expect_HCCL_E_INTERNAL)
{
    MockUnfoldDfxEnv();
    g_dfxRegisterRet = 1;
    ASSERT_EQ(threadManager->HcclDedicatedThreadDfx(MakeUnfoldDfxCallback()), HCCL_SUCCESS);
    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// AICPU_LAUNCH复用：二次获取同句柄、不二次分配、不重复注册DFX
TEST_F(ThreadManagerTest, Ut_UnfoldAcquire_When_Reacquire_Expect_ReuseWithoutDfxReRegister)
{
    MockUnfoldDfxEnv();
    ASSERT_EQ(threadManager->HcclDedicatedThreadDfx(MakeUnfoldDfxCallback()), HCCL_SUCCESS);
    ThreadHandle thread1 = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread1);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    ThreadHandle thread2 = 0;
    ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread2, thread1);           // 复用同句柄
    EXPECT_EQ(g_unfoldAllocCallCount, 1U); // 未发生第二次分配
    EXPECT_EQ(g_dfxRegisterCallCount, 1U); // 复用路径不重复注册
}

// 未记录即获取：首建点直接致命报错，判空前置于分配、零副作用
TEST_F(ThreadManagerTest, Ut_UnfoldAcquire_When_NoRecord_Expect_FatalError)
{
    MockUnfoldDfxEnv();
    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(g_unfoldAllocCallCount, 0U);
}

// 报错后补记录再重试：恢复正常，首建注册成功（验证零副作用可恢复）
TEST_F(ThreadManagerTest, Ut_UnfoldAcquire_When_RecordAfterError_Expect_RetrySuccess)
{
    MockUnfoldDfxEnv();
    ThreadHandle thread = 0;
    ASSERT_EQ(
        threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread), HCCL_E_INTERNAL);
    ASSERT_EQ(threadManager->HcclDedicatedThreadDfx(MakeUnfoldDfxCallback()), HCCL_SUCCESS);
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_unfoldAllocCallCount, 1U);
    EXPECT_EQ(g_dfxRegisterCallCount, 1U);
}

// 重复记录幂等：仅首次记录生效，首建只注册一次
TEST_F(ThreadManagerTest, Ut_UnfoldAcquire_When_RecordTwice_Expect_Idempotent)
{
    MockUnfoldDfxEnv();
    ASSERT_EQ(threadManager->HcclDedicatedThreadDfx(MakeUnfoldDfxCallback()), HCCL_SUCCESS);
    ASSERT_EQ(threadManager->HcclDedicatedThreadDfx(MakeUnfoldDfxCallback()), HCCL_SUCCESS);
    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_dfxRegisterCallCount, 1U);
}
