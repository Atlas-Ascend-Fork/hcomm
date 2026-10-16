/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "../../../hccl_api_base_test.h"
#include "thread_manager.h"
#include "dpu_thread_service.h"
#include "cpu_thread.h"
#include "thread.h" // hccl::Thread 完整类型 + LookupThreadByHandle（mock 目标）
#include "kernel_launch/dpu_taskexp_kernel_launch.h" // LaunchDpuTaskexpShmemRestoreKernel（mock 目标）
#include "hcomm_c_adpt.h"                            // HcommThreadAllocWithCommConfig（mock 目标）
#include "hcomm_res.h"                               // HcommThreadFree（mock 目标）
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "driver/ascend_hal.h"

using namespace hccl;
using Hccl::HrtGetDevicePhyIdByUserDevId;
using Hccl::HrtHalGetDeviceInfo;
using Hccl::HrtHalHostRegister;
using Hccl::HrtHalHostUnregister;
using Hccl::HrtResetXpuDevice;
using Hccl::HrtSetXpuDevice;

namespace {
constexpr uint8_t EXIT_SIGNAL = 3; // 对应 DEVICE_SIGNAL_THIRD，dpu 线程回写的退出信号
constexpr ThreadHandle kFakeHandle = 0x7000;

// 句柄不落 g_ThreadMap（StubAlloc 不调 SaveThreads），线程对象由本文件静态量持有，生命周期完全可控
std::shared_ptr<CpuThread> g_injectedCpuThread;
HcclResult g_lookupRet = HCCL_SUCCESS;
uint32_t g_allocCallCount = 0;
HcclResult g_launchRet = HCCL_SUCCESS;
// ThreadMgr 下传的 kernel 二进制句柄：下发器被桩住时不会解引用，用假值即可
const aclrtBinHandle kFakeBinHandle = reinterpret_cast<aclrtBinHandle>(0x5a5a);
HcommResult g_allocRet = HCCL_SUCCESS;

HcommResult StubThreadAllocWithCommConfig(
    CommEngine engine, const char* commId, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
    ThreadHandle* threads)
{
    (void)engine;
    (void)commId;
    (void)threadNum;
    (void)type;
    (void)config;
    ++g_allocCallCount;
    if (threads != nullptr) {
        *threads = kFakeHandle;
    }
    return g_allocRet;
}

HcclResult StubLookup(ThreadHandle handle, std::shared_ptr<Thread>& outThread)
{
    (void)handle;
    if (g_lookupRet == HCCL_SUCCESS) {
        outThread = g_injectedCpuThread;
    }
    return g_lookupRet;
}

// 桩 acquire/free/反查三个跨层依赖，让 EnsureDpuKernelLaunched 的查链命中与回滚路径可被精确驱动
void MockThreadMgrDeps()
{
    MOCKER(HcommThreadAllocWithCommConfig).stubs().will(invoke(StubThreadAllocWithCommConfig));
    MOCKER(HcommThreadFree).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(LookupThreadByHandle).stubs().will(invoke(StubLookup));
}

// taskexp kernel 下发器（L2 实现，与 DpuThreadService 同编进 hccl_llt）：g_launchRet 控制成功/失败。
// 不桩时走真实实现（binHandle 为空即 HCCL_E_PTR），用于验证 binHandle 一路透传。
void MockDpuTaskexpKernelLaunch() { MOCKER(LaunchDpuTaskexpShmemRestoreKernel).stubs().will(returnValue(g_launchRet)); }

// 复制 test_cpu_thread_class.cc 的 Init 依赖桩：让 legacy->Init() 走完 InitShmem + CreateTaskService 两步；
// UB 分支用真实 malloc，避免 PCIE 假地址被 memset_s 打崩
void MockInitDeps(DevType devType)
{
    static char realPathBuf[1] = {0};
    MOCKER(realpath).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(static_cast<char*>(realPathBuf)));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtGetCurrentContext).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryLoadFromFile)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtCreateStreamWithConfig)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryGetFunction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));

    int32_t logicDevId = 0;
    MOCKER(aclrtGetLogicDevIdByUserDevId)
        .stubs()
        .with(mockcpp::any(), outBoundP(&logicDevId))
        .will(returnValue(ACL_SUCCESS));
    int64_t connectType = HOST_DEVICE_CONNECT_TYPE_UB;
    MOCKER(HrtHalGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connectType))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtHalHostRegister).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtHalHostUnregister).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<Hccl::DevId>(0)));
}

ManagerCallbacks MakeCallbacks()
{
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
    return callbacks;
}

// 池查询断言收口：GetCpuThreadHandle 只按 (CPU, CPU) 查池，命中与否即「池里有无本 comm 的 DPU 线程」
bool PoolHasHostCpuThread(ThreadMgr& mgr)
{
    ThreadHandle handle = 0;
    return mgr.GetCpuThreadHandle(handle) == HCCL_SUCCESS;
}

// 已拉起（Init 成功）线程的清理：先写退出信号防 WaitKernelTerminate 自旋，再 DeInit，最后释放注入线程
void CleanupLaunched()
{
    LegacyDpuThread* legacy = g_injectedCpuThread->GetLegacy();
    *static_cast<uint8_t*>(legacy->GetDputagAddr()) = EXIT_SIGNAL;
    EXPECT_EQ(HCCL_SUCCESS, legacy->DeInit());
    g_injectedCpuThread.reset();
}
} // namespace

class TestDpuThreadService : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // 同 test_cpu_thread_class.cc：清掉 main.cc 预置的非法 HCCL_DFS_CONFIG，避免 LaunchKernel 抛异常
        unsetenv("HCCL_DFS_CONFIG");
        LegacyDpuThread::commNum_ = 0;
        g_injectedCpuThread.reset();
        g_lookupRet = HCCL_SUCCESS;
        g_allocCallCount = 0;
        g_launchRet = HCCL_SUCCESS;
        g_allocRet = HCCL_SUCCESS;
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_GetThreadFail_Expect_Rollback)
{
    MockThreadMgrDeps();
    g_lookupRet = HCCL_E_NOT_FOUND; // acquire 后 LookupThreadByHandle 桩返回失败，触发首条回滚路径

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    bool newCreated = true;
    HcclResult ret = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(nullptr, 0, &newCreated);

    EXPECT_EQ(HCCL_E_NOT_FOUND, ret);
    EXPECT_FALSE(newCreated);
    // 回滚已把池条目摘除、记账归零：查池落空，后续可重新 acquire 拉起
    EXPECT_FALSE(PoolHasHostCpuThread(mgr));
    EXPECT_EQ(0u, mgr.totalThreadCount_);
    EXPECT_EQ(0u, mgr.usedNotifyNum_);
    EXPECT_EQ(1u, g_allocCallCount);
}

TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_InitFail_Expect_Rollback)
{
    MockThreadMgrDeps();
    g_injectedCpuThread = std::make_shared<CpuThread>();
    g_lookupRet = HCCL_SUCCESS;

    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_RUNTIME));

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    bool newCreated = false;
    HcclResult ret = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(nullptr, 0, &newCreated);

    EXPECT_EQ(HCCL_E_INTERNAL, ret); // legacy Init 在 HrtSetXpuDevice 失败
    EXPECT_FALSE(newCreated);
    EXPECT_FALSE(PoolHasHostCpuThread(mgr));
    EXPECT_EQ(0u, mgr.totalThreadCount_);
    EXPECT_EQ(0u, mgr.usedNotifyNum_);

    // Init 半途失败，kernel 未拉起：注入线程析构即回收，无需 DeInit
    g_injectedCpuThread.reset();
}

TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_LaunchFail_Expect_Rollback)
{
    MockThreadMgrDeps();
    g_injectedCpuThread = std::make_shared<CpuThread>();
    g_lookupRet = HCCL_SUCCESS;
    g_launchRet = HCCL_E_INTERNAL; // 强桩注入 launch 失败
    MockDpuTaskexpKernelLaunch();

    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    bool newCreated = false;
    // commV2 非空才走 launch 分支（950 语义）
    HcclResult ret
        = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(reinterpret_cast<HcclComm>(0x1), 0, &newCreated);

    EXPECT_EQ(HCCL_E_INTERNAL, ret);
    EXPECT_FALSE(newCreated);
    EXPECT_FALSE(PoolHasHostCpuThread(mgr));
    EXPECT_EQ(0u, mgr.totalThreadCount_);
    EXPECT_EQ(0u, mgr.usedNotifyNum_);

    CleanupLaunched();
}

TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_950LaunchSuccess_Expect_Success)
{
    MockThreadMgrDeps();
    g_injectedCpuThread = std::make_shared<CpuThread>();
    g_lookupRet = HCCL_SUCCESS;
    g_launchRet = HCCL_SUCCESS;
    MockDpuTaskexpKernelLaunch();

    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    bool newCreated = false;
    // commV2 非空：走 950 的 taskexp kernel 下发分支
    HcclResult ret
        = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(reinterpret_cast<HcclComm>(0x1), 0, &newCreated);

    EXPECT_EQ(HCCL_SUCCESS, ret);
    EXPECT_TRUE(newCreated);
    EXPECT_TRUE(PoolHasHostCpuThread(mgr));
    EXPECT_TRUE(g_injectedCpuThread->GetLegacy()->IsKernelLaunched());

    CleanupLaunched();
}

TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_Normal_Expect_Success_And_Idempotent)
{
    MockThreadMgrDeps();
    g_injectedCpuThread = std::make_shared<CpuThread>();
    g_lookupRet = HCCL_SUCCESS;
    g_launchRet = HCCL_SUCCESS;

    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    bool newCreated = false;
    // commV2 = nullptr（910 语义）：跳过 launch，仅完成三步收编
    HcclResult ret = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(nullptr, 0, &newCreated);

    EXPECT_EQ(HCCL_SUCCESS, ret);
    EXPECT_TRUE(newCreated);
    EXPECT_TRUE(PoolHasHostCpuThread(mgr));

    // 幂等：二次调用查链命中（已拉起），不再触发 acquire
    uint32_t allocBefore = g_allocCallCount;
    bool newCreated2 = true;
    HcclResult ret2 = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(nullptr, 0, &newCreated2);
    EXPECT_EQ(HCCL_SUCCESS, ret2);
    EXPECT_FALSE(newCreated2);
    EXPECT_EQ(allocBefore, g_allocCallCount);

    ThreadHandle got = 0;
    EXPECT_EQ(HCCL_SUCCESS, mgr.GetCpuThreadHandle(got));
    EXPECT_EQ(kFakeHandle, got);

    CleanupLaunched();
}

TEST_F(TestDpuThreadService, Ut_RollbackThread_When_Acquired_Expect_AccountingRestored)
{
    MockThreadMgrDeps();

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    ThreadConfig config;
    ASSERT_EQ(HCCL_SUCCESS, ThreadConfigInit(&config, 1));
    config.notifyNumPerThread = 3;

    ThreadHandle handle = 0;
    std::vector<uint32_t> tid;
    ASSERT_EQ(
        HCCL_SUCCESS,
        mgr.HcclThreadAcquireV2(CommEngine::COMM_ENGINE_CPU, 1, ThreadType::THREAD_TYPE_CPU, &config, &handle, tid));
    EXPECT_EQ(kFakeHandle, handle);
    EXPECT_EQ(1u, mgr.totalThreadCount_);
    EXPECT_EQ(3u, mgr.usedNotifyNum_);

    EXPECT_EQ(HCCL_SUCCESS, mgr.RollbackThread(handle));
    EXPECT_EQ(0u, mgr.totalThreadCount_);
    EXPECT_EQ(0u, mgr.usedNotifyNum_);

    EXPECT_FALSE(PoolHasHostCpuThread(mgr));
}

TEST_F(TestDpuThreadService, Ut_RollbackThread_When_NotInPool_Expect_NotFound)
{
    MockThreadMgrDeps();

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    EXPECT_EQ(HCCL_E_NOT_FOUND, mgr.RollbackThread(0x9999));
}

TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_DevIdUnset_Expect_Para)
{
    MockThreadMgrDeps();

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    bool newCreated = true;
    // UINT32_MAX 是「未设置」哨兵：须在 acquire 之前判参，避免下游以 0xFFFFFFFF 失败
    HcclResult ret = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(nullptr, UINT32_MAX, &newCreated);

    EXPECT_EQ(HCCL_E_PARA, ret);
    EXPECT_FALSE(newCreated);
    EXPECT_EQ(0u, g_allocCallCount); // 判参先于 acquire：未产生任何分配
    EXPECT_FALSE(PoolHasHostCpuThread(mgr));
}

TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_AcquireFail_Expect_Propagate)
{
    MockThreadMgrDeps();
    g_allocRet = static_cast<HcommResult>(HCCL_E_UNAVAIL); // 强桩注入 acquire 失败

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    bool newCreated = true;
    HcclResult ret = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(nullptr, 0, &newCreated);

    EXPECT_EQ(HCCL_E_UNAVAIL, ret);
    EXPECT_FALSE(newCreated);
    EXPECT_EQ(1u, g_allocCallCount);
    // 分配失败线程未入池，无残留记账
    EXPECT_EQ(0u, mgr.totalThreadCount_);
    EXPECT_EQ(0u, mgr.usedNotifyNum_);
    EXPECT_FALSE(PoolHasHostCpuThread(mgr));
}

TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_HalfLaunched_Expect_Internal)
{
    MockThreadMgrDeps();
    // 池中已有条目（acquire 已登记），但该 CpuThread 未过 Init：TaskService/DPUTAG 均为空
    g_injectedCpuThread = std::make_shared<CpuThread>();
    g_lookupRet = HCCL_SUCCESS;

    ThreadMgr mgr(4, 8, "testComm", kFakeBinHandle, MakeCallbacks());
    ThreadConfig config;
    ASSERT_EQ(HCCL_SUCCESS, ThreadConfigInit(&config, 1));
    ThreadHandle handle = 0;
    std::vector<uint32_t> tid;
    ASSERT_EQ(
        HCCL_SUCCESS,
        mgr.HcclThreadAcquireV2(CommEngine::COMM_ENGINE_CPU, 1, ThreadType::THREAD_TYPE_CPU, &config, &handle, tid));

    uint32_t allocBefore = g_allocCallCount;
    bool newCreated = true;
    HcclResult ret = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(nullptr, 0, &newCreated);

    // 半拉起状态显式报错：不伪装成功（下游会把未拉起的 kernel 当已拉起用），也不再 acquire
    EXPECT_EQ(HCCL_E_INTERNAL, ret);
    EXPECT_FALSE(newCreated);
    EXPECT_EQ(allocBefore, g_allocCallCount);
    EXPECT_EQ(1u, mgr.totalThreadCount_); // 池条目归属未变，未被本次调用回滚

    mgr.RollbackThread(handle);
    g_injectedCpuThread.reset();
}

// 下发器不桩（走真实实现）：binHandle 为空时应在它的第一道判参失败并被回滚。
// 这条用例同时是「ThreadMgr → DpuThreadService → 下发器」整条 binHandle 透传链的接线断言。
TEST_F(TestDpuThreadService, Ut_EnsureDpuKernelLaunched_When_BinHandleNull_Expect_PtrErrorAndRollback)
{
    MockThreadMgrDeps();
    g_injectedCpuThread = std::make_shared<CpuThread>();
    g_lookupRet = HCCL_SUCCESS;

    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    ThreadMgr mgr(4, 8, "testComm", nullptr, MakeCallbacks());
    bool newCreated = false;
    HcclResult ret
        = mgr.GetDpuThreadService()->EnsureDpuKernelLaunched(reinterpret_cast<HcclComm>(0x1), 0, &newCreated);

    EXPECT_EQ(HCCL_E_PTR, ret); // 真实下发器的首道判参：binHandle 为空
    EXPECT_FALSE(newCreated);
    EXPECT_FALSE(PoolHasHostCpuThread(mgr));
    EXPECT_EQ(0u, mgr.totalThreadCount_);
    EXPECT_EQ(0u, mgr.usedNotifyNum_);

    CleanupLaunched();
}
