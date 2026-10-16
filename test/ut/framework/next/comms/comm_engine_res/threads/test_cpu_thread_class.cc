/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdexcept>
#include <vector>
#include "../../../hccl_api_base_test.h"
#include "cpu_thread.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "driver/ascend_hal.h"

using namespace hccl;
// Hrt* 系列在 namespace Hccl（extern "C" 链接），用目标 using 声明引入，
// 避免 `using namespace Hccl;` 与 hccl 命名空间产生重名二义
using Hccl::HrtGetDevicePhyIdByUserDevId;
using Hccl::HrtHalGetDeviceInfo;
using Hccl::HrtHalHostRegister;
using Hccl::HrtHalHostUnregister;
using Hccl::HrtResetXpuDevice;
using Hccl::HrtSetXpuDevice;

class TestCpuThread : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // main.cc 预置了非法的 HCCL_DFS_CONFIG=connection_fault_detction_time:0，
        // LaunchKernel 经 EnvConfig::GetInstance() 读到会抛异常，须先清除（同 test_thread_api.cc）
        unsetenv("HCCL_DFS_CONFIG");
        // commNum_ 是进程级静态量，且按 legacy 记账语义「末次」走 reset 分支时不自减，
        // 计数会残留。不清零会让用例互相干扰（UT 目标带 -fno-access-control，可访问私有成员）
        LegacyDpuThread::commNum_ = 0;
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

namespace {
constexpr uint8_t EXIT_SIGNAL = 3; // 对应 DEVICE_SIGNAL_THIRD，dpu 线程回写的退出信号

// Init 失败矩阵的注入点。mockcpp「先注册先命中」，同一依赖一旦被 MockInitDeps 注册成功桩，
// 后面再注册失败桩不会生效，故失败必须在注册时就选定，不能在用例里叠加。
enum class InitFailPoint {
    kNone = 0,
    kGetDeviceType,
    kLogicDevId,
    kGetCurrentContextNpu,
    kGetCurrentContextDpu,
    kBinaryLoad,
    kCreateStream,
    kGetFunction,
    kSetContextNpuAfterLaunch, // kernel 已下发成功后，切回 npu ctx 失败
    kDepThrows,                // 链上依赖抛异常（异常边界用例）
};

// aclrtGetCurrentContext 在 Init 内被调用两次（先 npu ctx、后 dpu ctx），两次签名相同，
// 失败注入须按调用次序区分，故用计数桩而非固定返回值桩
uint32_t g_getCtxCall = 0;
uint32_t g_getCtxFailAt = 0;
aclError StubGetCurrentContext(aclrtContext* ctx)
{
    ++g_getCtxCall;
    if (ctx != nullptr) {
        *ctx = reinterpret_cast<aclrtContext>(0x1000);
    }
    return (g_getCtxFailAt == g_getCtxCall) ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

// DeInit 内 aclrtSetCurrentContext 同样是两次（切 dpu ctx 销毁流、切回 npu ctx），同款计数桩
uint32_t g_setCtxCall = 0;
uint32_t g_setCtxFailAt = 0;
aclError StubSetCurrentContext(aclrtContext ctx)
{
    (void)ctx;
    ++g_setCtxCall;
    return (g_setCtxFailAt == g_setCtxCall) ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

// DeInit 错误分支的准备：不跑完整 Init，直接置私有成员（UT 目标带 -fno-access-control），
// 让终止握手第一次读即命中退出信号，从而只观测流销毁/XPU 记账/上下文还原各步的失败处理
uint8_t g_signalBuf[8] = {0};
void PrepareLaunchedState(LegacyDpuThread* legacy)
{
    g_signalBuf[0] = EXIT_SIGNAL;
    legacy->isKernelLaunched_ = true;
    legacy->dputagHostPtr_ = g_signalBuf;
    legacy->rtStream_ = reinterpret_cast<rtStream_t>(0x1000); // 假流句柄：销毁动作已被桩接住
}

// 失败出口回收已建流的观测点：假句柄由 StubCreateStreamWithFakeHandle 建出，
// 断言被回收的正是这条流、且只回收一次（成功后 DeInit 不得二次销毁）
constexpr uint64_t FAKE_STREAM = 0x2000;
aclError StubCreateStreamWithFakeHandle(aclrtStream* stream, uint32_t priority, uint32_t flag)
{
    (void)priority;
    (void)flag;
    if (stream != nullptr) {
        *stream = reinterpret_cast<aclrtStream>(FAKE_STREAM);
    }
    return ACL_SUCCESS;
}

uint32_t g_destroyStreamCount = 0;
uint32_t g_destroyedStreamMatched = 0;
aclError StubDestroyStreamForce(aclrtStream stream)
{
    ++g_destroyStreamCount;
    g_destroyedStreamMatched = (stream == reinterpret_cast<aclrtStream>(FAKE_STREAM)) ? 1u : 0u;
    return ACL_SUCCESS;
}

// Init 成功路径上除 hrtGetDeviceType / HrtSetXpuDevice / HrtResetXpuDevice 以外的全部依赖。
// 注意：
//  1. aclrtCreateStreamWithConfig 默认不注入 fake stream，让 rtStream_ 保持 nullptr；
//     需要观测「失败出口回收已建流」的用例传 fakeStream = true，由该桩回填假句柄；
//  2. HrtHalGetDeviceInfo 返回 UB 连接类型，让 DpuShmemMgr::Alloc 走「真实 malloc + 4K 对齐」分支，
//     避免 PCIE 分支 HrtMalloc 桩返回假地址 0x12345678 被 memset_s 写入导致段错误；
//  3. fail 指定让哪一个依赖失败，其余保持成功，用于逐步覆盖 Init 的失败矩阵。
void MockInitDeps(DevType devType, InitFailPoint fail = InitFailPoint::kNone, bool fakeStream = false)
{
    static char realPathBuf[1] = {0};
    g_getCtxCall = 0;
    g_getCtxFailAt = (fail == InitFailPoint::kGetCurrentContextNpu) ?
                         1u :
                         ((fail == InitFailPoint::kGetCurrentContextDpu) ? 2u : 0u);
    MOCKER(realpath).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(static_cast<char*>(realPathBuf)));
    MOCKER(hrtGetDeviceType)
        .stubs()
        .with(outBound(devType))
        .will(returnValue(fail == InitFailPoint::kGetDeviceType ? HCCL_E_RUNTIME : HCCL_SUCCESS));
    if (fail == InitFailPoint::kDepThrows) {
        // 异常边界用例：共享内存与 TaskService 都已就绪后抛异常，验证 Init 把异常转成错误码交给调用方回滚，
        // 而不是穿出 HcclTaskRegister 的 extern "C" 边界（穿出即 terminate）
        MOCKER(aclrtGetCurrentContext).stubs().will(throws(std::runtime_error("stub throws")));
    } else {
        MOCKER(aclrtGetCurrentContext).stubs().with(mockcpp::any()).will(invoke(StubGetCurrentContext));
    }
    // aclrtSetCurrentContext 在 Init 路径上首次被调用即 LaunchKernel 里「切回 npu ctx」，故用计数桩按次注入；
    // failAt == 0 时恒成功，与固定成功桩等价
    g_setCtxCall = 0;
    g_setCtxFailAt = (fail == InitFailPoint::kSetContextNpuAfterLaunch) ? 1u : 0u;
    MOCKER(aclrtSetCurrentContext).stubs().with(mockcpp::any()).will(invoke(StubSetCurrentContext));
    MOCKER(aclrtBinaryLoadFromFile)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fail == InitFailPoint::kBinaryLoad ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS));
    if (fakeStream) {
        MOCKER(aclrtCreateStreamWithConfig)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(StubCreateStreamWithFakeHandle));
    } else {
        MOCKER(aclrtCreateStreamWithConfig)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(fail == InitFailPoint::kCreateStream ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS));
    }
    MOCKER(aclrtBinaryGetFunction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fail == InitFailPoint::kGetFunction ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS));

    // InitShmem 的 DPU 共享内存申请链（DPUTAG + DPUTASKEXCEPTION）
    int32_t logicDevId = 0;
    MOCKER(aclrtGetLogicDevIdByUserDevId)
        .stubs()
        .with(mockcpp::any(), outBoundP(&logicDevId))
        .will(returnValue(fail == InitFailPoint::kLogicDevId ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS));
    int64_t connectType = HOST_DEVICE_CONNECT_TYPE_UB;
    MOCKER(HrtHalGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connectType))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtHalHostRegister).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtHalHostUnregister).stubs().will(returnValue(HCCL_SUCCESS));

    // Stream(nullptr) 构造里 InitDevPhyId 的依赖
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<Hccl::DevId>(0)));
}

// Init 完整成功路径（SetTaskServiceArgs + LegacyDpuThread::Init，内部已含 LaunchKernel），
// 并预置退出信号防 WaitKernelTerminate 自旋等待
HcclResult InitAndLaunch(CpuThread& thread)
{
    Hccl::TaskServiceArgs args{"testCommId", 0};
    thread.GetLegacy()->SetTaskServiceArgs(args);
    HcclResult ret = thread.GetLegacy()->Init();
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    // 终止握手第一次读就要读到 DEVICE_SIGNAL_THIRD，否则会自旋等待
    *static_cast<uint8_t*>(thread.GetLegacy()->GetDputagAddr()) = EXIT_SIGNAL;
    return HCCL_SUCCESS;
}

// HrtResetXpuDevice 调用计数：mockcpp 的 expects(never()) 会「霸占」同一 MOCKER 后续注册的
// once() 调用链（先注册先命中），无法表达「前半段不调用、后半段恰好一次」，
// 故用 invoke 计数断言「先销毁不 reset、末次才 reset」
uint32_t g_resetXpuCount = 0;
HcclResult StubResetXpuDevice(uint32_t devType, const uint32_t devId)
{
    (void)devType;
    (void)devId;
    ++g_resetXpuCount;
    return HCCL_SUCCESS;
}
} // namespace

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(invoke(StubResetXpuDevice));

    CpuThread thread;
    EXPECT_EQ(HCCL_SUCCESS, InitAndLaunch(thread));
    EXPECT_NE(nullptr, thread.GetStream());
    EXPECT_TRUE(thread.GetLegacy()->IsKernelLaunched());
    // HOST CPU 线程不占 local notify，GetNotify 恒为空
    EXPECT_EQ(0u, thread.GetNotifyNum());
    EXPECT_EQ(nullptr, thread.GetNotify(0));
    EXPECT_EQ(HCCL_SUCCESS, thread.SupplementNotify(3));
    EXPECT_NO_THROW(thread.LaunchTask());
    // 手动 DeInit 完成 teardown：析构前把 isKernelLaunched_ 清掉，避免 StopTaskexpService 自旋等待
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
}

// Init 链上依赖抛异常（STL 分配/Esc 解析/实例构造等）时不得穿出 Init：调用者 HcclTaskRegister 是
// extern "C" 边界，异常穿出即 terminate；须转成错误码，交由调用方的 RollbackAcquiredThread 回滚
TEST_F(TestCpuThread, Ut_CpuThread_Init_When_DepThrows_Expect_ErrorNotPropagate)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kDepThrows);

    CpuThread thread;
    EXPECT_EQ(HCCL_E_INTERNAL, InitAndLaunch(thread)); // 异常被拦下转成错误码，本行不抛
    EXPECT_FALSE(thread.GetLegacy()->IsKernelLaunched());
}

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_SetXpuDeviceFailed_Expect_Return_HCCL_E_INTERNAL)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_RUNTIME));
    MOCKER(HrtResetXpuDevice).expects(never());

    CpuThread thread;
    Hccl::TaskServiceArgs args{"testCommId", 0};
    thread.GetLegacy()->SetTaskServiceArgs(args);
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetLegacy()->Init());
    // Init 半途失败时 kernel 未拉起，DeInit 应早退且不 reset
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
}

TEST_F(TestCpuThread, Ut_CpuThread_DeInit_When_NotLaunched_Expect_Success_And_NoReset)
{
    MOCKER(HrtResetXpuDevice).expects(never());

    CpuThread thread;
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit()); // 幂等
}

TEST_F(TestCpuThread, Ut_CpuThread_DeInit_When_Launched_Expect_ResetOnce)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));

    CpuThread thread;
    EXPECT_EQ(HCCL_SUCCESS, InitAndLaunch(thread));

    // 此时 commNum_ == 1，走 reset 分支（legacy 语义）
    MOCKER(HrtResetXpuDevice).expects(once()).with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
    // 二次 DeInit 幂等：不再 reset
    MOCKER(HrtResetXpuDevice).expects(never());
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
}

TEST_F(TestCpuThread, Ut_CpuThread_DeInit_When_TwoCommsAlive_Expect_ResetOnLastOnly)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));

    CpuThread threadA;
    CpuThread threadB;
    EXPECT_EQ(HCCL_SUCCESS, InitAndLaunch(threadA));
    EXPECT_EQ(HCCL_SUCCESS, InitAndLaunch(threadB));

    g_resetXpuCount = 0;
    MOCKER(HrtResetXpuDevice).stubs().will(invoke(StubResetXpuDevice));

    // 两个对象同时存活：commNum_ == 2，先销毁的那个只自减，不 reset
    EXPECT_EQ(HCCL_SUCCESS, threadA.GetLegacy()->DeInit());
    EXPECT_EQ(0u, g_resetXpuCount);

    // 末次销毁才 reset
    EXPECT_EQ(HCCL_SUCCESS, threadB.GetLegacy()->DeInit());
    EXPECT_EQ(1u, g_resetXpuCount);
}

TEST_F(TestCpuThread, Ut_CpuThread_LaunchKernel_When_LaunchFailed_Expect_Return_HCCL_E_INTERNAL)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_ERROR_INVALID_PARAM));

    CpuThread thread;
    // 下发失败发生在 Init 内部的 LaunchKernel 步骤
    EXPECT_EQ(HCCL_E_INTERNAL, InitAndLaunch(thread));

    // 下发失败 → isKernelLaunched_ 仍为假 → DeInit 早退且不 reset
    MOCKER(HrtResetXpuDevice).expects(never());
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
}

// kernel 已下发成功、随后切回 npu ctx 失败：kernel 已在跑，标志必须已置位，否则析构链因 !isKernelLaunched_
// 短路，终止握手/流销毁/XPU reset 全被跳过，kernel 线程永不退出
TEST_F(TestCpuThread, Ut_CpuThread_LaunchKernel_When_SetNpuCtxAfterLaunchFailed_Expect_LaunchedAndTeardown)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kSetContextNpuAfterLaunch);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    EXPECT_EQ(HCCL_E_INTERNAL, InitAndLaunch(thread));
    EXPECT_TRUE(thread.GetLegacy()->IsKernelLaunched());

    // InitAndLaunch 失败时不会预置退出信号，此处补齐以避免 WaitKernelTerminate 自旋；
    // 本用例未拉起 AICPU taskexception 监控，预置停止位避免 StopTaskexpService 空等 10s
    *static_cast<uint8_t*>(thread.GetLegacy()->GetDputagAddr()) = EXIT_SIGNAL;
    thread.GetLegacy()->taskExpStopped_ = true;
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    g_resetXpuCount = 0;
    MOCKER(HrtResetXpuDevice).stubs().will(invoke(StubResetXpuDevice));

    // teardown 未被短路：握手读到退出信号 → 流销毁 → 末次销毁走 XPU reset
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
    EXPECT_EQ(nullptr, thread.GetLegacy()->rtStream_);
    EXPECT_EQ(1u, g_resetXpuCount);
}

TEST_F(TestCpuThread, Ut_CpuThread_LaunchKernel_When_TaskServiceNull_Expect_Return_HCCL_E_INTERNAL)
{
    CpuThread thread;
    // 未 Init（taskService_ 为空），直接下发 kernel 应失败
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetLegacy()->LaunchKernel());
}

TEST_F(TestCpuThread, Ut_CpuThread_IsDeviceA5_When_A5Device_Expect_True)
{
    CpuThread thread;
    // 直接置 devType_（-fno-access-control 可访问私有成员），避免走完整 Init 收编三步
    thread.GetLegacy()->devType_ = DevType::DEV_TYPE_950;
    EXPECT_TRUE(thread.IsDeviceA5());
}

TEST_F(TestCpuThread, Ut_CpuThread_IsDeviceA5_When_A3Device_Expect_False)
{
    CpuThread thread;
    thread.GetLegacy()->devType_ = DevType::DEV_TYPE_910;
    EXPECT_FALSE(thread.IsDeviceA5());
}

TEST_F(TestCpuThread, Ut_CpuThread_LocalPlaneOps_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    CpuThread thread;
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, thread.LocalNotifyRecord(0U));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, thread.LocalNotifyWait(0U));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, thread.LocalNotifyRecord(static_cast<ThreadHandle>(0), 0U));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, thread.LocalNotifyWait(0U, 1000U));

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, thread.LocalCopy(dst, src, 1024));
    EXPECT_EQ(
        HCCL_E_NOT_SUPPORT,
        thread.LocalReduce(dst, src, 1024, HcommDataType::HCOMM_DATA_TYPE_INT32, HcommReduceOp::HCOMM_REDUCE_SUM));
}

// ---------------- Init 失败矩阵：逐个依赖返回失败并断言对应错误码 ----------------

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_GetDeviceTypeFailed_Expect_E_RUNTIME)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kGetDeviceType);
    MOCKER(HrtSetXpuDevice).expects(never()); // 首个依赖即失败，未切入 dpu 域

    CpuThread thread;
    thread.GetLegacy()->SetTaskServiceArgs(Hccl::TaskServiceArgs{"testCommId", 0});
    EXPECT_EQ(HCCL_E_RUNTIME, thread.GetLegacy()->Init());
    EXPECT_FALSE(thread.GetLegacy()->IsKernelLaunched());
}

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_LogicDevIdFailed_Expect_E_RUNTIME)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kLogicDevId);
    MOCKER(HrtSetXpuDevice).expects(never());

    CpuThread thread;
    thread.GetLegacy()->SetTaskServiceArgs(Hccl::TaskServiceArgs{"testCommId", 0});
    // InitShmem 第一步失败：DPUTAG / DPUTASKEXCEPTION 都未申请，无内存泄漏面
    EXPECT_EQ(HCCL_E_RUNTIME, thread.GetLegacy()->Init());
    EXPECT_EQ(nullptr, thread.GetLegacy()->GetDputagAddr());
}

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_GetNpuCtxFailed_Expect_HCCL_E_INTERNAL)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kGetCurrentContextNpu);
    MOCKER(HrtSetXpuDevice).expects(never());

    CpuThread thread;
    thread.GetLegacy()->SetTaskServiceArgs(Hccl::TaskServiceArgs{"testCommId", 0});
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetLegacy()->Init());
    EXPECT_FALSE(thread.GetLegacy()->IsKernelLaunched());
}

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_GetDpuCtxFailed_Expect_HCCL_E_INTERNAL)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kGetCurrentContextDpu);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtResetXpuDevice).expects(never()); // 失败半途未拉起，析构不 reset

    CpuThread thread;
    thread.GetLegacy()->SetTaskServiceArgs(Hccl::TaskServiceArgs{"testCommId", 0});
    // 已在 dpu 域内失败：由 DpuCtxRollback 守卫切回 npu ctx（aclrtSetCurrentContext 已桩成功）
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetLegacy()->Init());
    EXPECT_FALSE(thread.GetLegacy()->IsKernelLaunched());
}

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_BinaryLoadFailed_Expect_OpenFileFailure)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kBinaryLoad);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    thread.GetLegacy()->SetTaskServiceArgs(Hccl::TaskServiceArgs{"testCommId", 0});
    EXPECT_EQ(HCCL_E_OPEN_FILE_FAILURE, thread.GetLegacy()->Init());
    EXPECT_FALSE(thread.GetLegacy()->IsKernelLaunched());
    EXPECT_EQ(nullptr, thread.GetStream()); // 未走到建流
}

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_CreateStreamFailed_Expect_HCCL_E_INTERNAL)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kCreateStream);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    thread.GetLegacy()->SetTaskServiceArgs(Hccl::TaskServiceArgs{"testCommId", 0});
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetLegacy()->Init());
    EXPECT_FALSE(thread.GetLegacy()->IsKernelLaunched());
    EXPECT_EQ(nullptr, thread.GetStream());
}

TEST_F(TestCpuThread, Ut_CpuThread_Init_When_GetFunctionFailed_Expect_HCCL_E_INTERNAL)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kGetFunction);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    thread.GetLegacy()->SetTaskServiceArgs(Hccl::TaskServiceArgs{"testCommId", 0});
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetLegacy()->Init());
    EXPECT_FALSE(thread.GetLegacy()->IsKernelLaunched());
}

// 失败时「流已建、kernel 未下发」：DeInit 因 isKernelLaunched_ 为 false 整体短路，Stream 又不是
// stream owner，不显式回收则半拉起线程被回滚销毁后每次重试都漏一条 dpu 流
TEST_F(TestCpuThread, Ut_CpuThread_Init_When_GetFunctionFailedAfterStreamCreated_Expect_StreamRecycled)
{
    MockInitDeps(DevType::DEV_TYPE_950, InitFailPoint::kGetFunction, true);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    g_destroyStreamCount = 0;
    g_destroyedStreamMatched = 0;
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(invoke(StubDestroyStreamForce));

    CpuThread thread;
    thread.GetLegacy()->SetTaskServiceArgs(Hccl::TaskServiceArgs{"testCommId", 0});
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetLegacy()->Init());
    EXPECT_FALSE(thread.GetLegacy()->IsKernelLaunched());
    // 假流句柄确已建出（否则本用例无意义），且被失败出口按同一句柄回收、句柄已置空
    EXPECT_EQ(1u, g_destroyStreamCount);
    EXPECT_EQ(1u, g_destroyedStreamMatched);
    EXPECT_EQ(nullptr, thread.GetLegacy()->rtStream_);
    EXPECT_EQ(nullptr, thread.GetStream());
    // DeInit 在未拉起时早退，不得二次销毁同一条流
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
    EXPECT_EQ(1u, g_destroyStreamCount);
}

// ---------------- WaitKernelTerminate：判空 / 握手读写失败 / 无回写超时 ----------------

TEST_F(TestCpuThread, Ut_CpuThread_WaitKernelTerminate_When_DputagPtrNull_Expect_HCCL_E_PTR)
{
    CpuThread thread;
    // InitShmem 未完成（dputagHostPtr_ 为空）：无从读写握手位，直接报指针错
    EXPECT_EQ(HCCL_E_PTR, thread.GetLegacy()->WaitKernelTerminate());
}

TEST_F(TestCpuThread, Ut_CpuThread_WaitKernelTerminate_When_MemcpyFailed_Expect_HCCL_E_INTERNAL)
{
    MOCKER(memcpy_s).stubs().will(returnValue(1)); // 1 != EOK

    CpuThread thread;
    thread.GetLegacy()->dputagHostPtr_ = g_signalBuf;
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetLegacy()->WaitKernelTerminate());
}

TEST_F(TestCpuThread, Ut_CpuThread_WaitKernelTerminate_When_NoResponse_Expect_E_TIMEOUT)
{
    // 不预置退出信号：device 侧无人回写 3，须在上限（WAIT_CMD_TIMEOUT = 10s）后返回超时。
    // 注意本用例真实耗时约 10s
    g_signalBuf[0] = 0;
    CpuThread thread;
    thread.GetLegacy()->dputagHostPtr_ = g_signalBuf;
    EXPECT_EQ(HCCL_E_TIMEOUT, thread.GetLegacy()->WaitKernelTerminate());
    EXPECT_EQ(2u, g_signalBuf[0]); // 已按协议写入 DEVICE_SIGNAL_SECOND
}

// ---------------- StopTaskexpService：短路 / 握手读写失败 / 无回执超时 ----------------

TEST_F(TestCpuThread, Ut_CpuThread_StopTaskexpService_When_NoTaskExpMem_Expect_Success)
{
    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    // kernel 已拉起但异常区未申请（910 无 taskexception）：无监控可停，不得空转握手
    legacy->isKernelLaunched_ = true;
    EXPECT_EQ(HCCL_SUCCESS, legacy->StopTaskexpService());
    legacy->isKernelLaunched_ = false;
}

TEST_F(TestCpuThread, Ut_CpuThread_StopTaskexpService_When_AlreadyStopped_Expect_Success)
{
    static uint8_t expBuf[8] = {0};
    expBuf[0] = 1; // 若真进入握手，1 不会被回写成 0，会走到 10s 超时并返回 E_TIMEOUT
    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    legacy->taskExpStopped_ = true;
    legacy->isKernelLaunched_ = true;
    legacy->taskExpMem_ = expBuf;
    EXPECT_EQ(HCCL_SUCCESS, legacy->StopTaskexpService());
    legacy->isKernelLaunched_ = false;
}

TEST_F(TestCpuThread, Ut_CpuThread_StopTaskexpService_When_MemcpyFailed_Expect_HCCL_E_INTERNAL)
{
    MOCKER(memcpy_s).stubs().will(returnValue(1));

    static uint8_t expBuf[8] = {0};
    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    legacy->isKernelLaunched_ = true;
    legacy->taskExpMem_ = expBuf;
    EXPECT_EQ(HCCL_E_INTERNAL, legacy->StopTaskexpService());
    legacy->isKernelLaunched_ = false; // 防析构路径再次进入握手
}

TEST_F(TestCpuThread, Ut_CpuThread_StopTaskexpService_When_NoResponse_Expect_E_TIMEOUT)
{
    static uint8_t expBuf[8] = {0}; // device 侧无人回写 0：吃满 10s 上限后返回超时
    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    legacy->isKernelLaunched_ = true;
    legacy->taskExpMem_ = expBuf;
    EXPECT_EQ(HCCL_E_TIMEOUT, legacy->StopTaskexpService());
    legacy->isKernelLaunched_ = false;
}

// ---------------- DeInit：每一步失败都记首个错误码，且后续步骤必须走完 ----------------

TEST_F(TestCpuThread, Ut_CpuThread_DeInit_When_WaitTerminateFailed_Expect_HCCL_E_PTR)
{
    MOCKER(aclrtSetCurrentContext).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    legacy->isKernelLaunched_ = true; // dputagHostPtr_ 故意留空：握手前置校验失败
    legacy->rtStream_ = reinterpret_cast<rtStream_t>(0x1000);
    // 握手失败不提前返回：流销毁 / XPU 记账 / 上下文还原仍走完，返回首个错误码
    EXPECT_EQ(HCCL_E_PTR, legacy->DeInit());
    EXPECT_EQ(nullptr, legacy->rtStream_);
}

TEST_F(TestCpuThread, Ut_CpuThread_DeInit_When_SetDpuCtxFailed_Expect_Runtime_And_NoDestroyStream)
{
    g_setCtxCall = 0;
    g_setCtxFailAt = 1; // 第一次切换（切 dpu ctx）失败
    MOCKER(aclrtSetCurrentContext).stubs().will(invoke(StubSetCurrentContext));
    MOCKER(aclrtDestroyStreamForce).expects(never()); // 不在 dpu ctx 内：不得销毁流
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    PrepareLaunchedState(legacy);
    EXPECT_EQ(HCCL_E_RUNTIME, legacy->DeInit());
}

TEST_F(TestCpuThread, Ut_CpuThread_DeInit_When_DestroyStreamFailed_Expect_Runtime)
{
    MOCKER(aclrtSetCurrentContext).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_ERROR_INVALID_PARAM));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    PrepareLaunchedState(legacy);
    EXPECT_EQ(HCCL_E_RUNTIME, legacy->DeInit());
}

TEST_F(TestCpuThread, Ut_CpuThread_DeInit_When_ResetXpuFailed_Expect_Propagate)
{
    MOCKER(aclrtSetCurrentContext).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_E_RUNTIME));

    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    PrepareLaunchedState(legacy);
    // commNum_ == 0（SetUp 已清零）→ 走末次销毁的 reset 分支
    EXPECT_EQ(HCCL_E_RUNTIME, legacy->DeInit());
    EXPECT_EQ(0u, LegacyDpuThread::commNum_);
}

TEST_F(TestCpuThread, Ut_CpuThread_DeInit_When_SetNpuCtxFailed_Expect_Runtime)
{
    g_setCtxCall = 0;
    g_setCtxFailAt = 2; // 第二次切换（切回 npu ctx）失败
    MOCKER(aclrtSetCurrentContext).stubs().will(invoke(StubSetCurrentContext));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    auto* legacy = thread.GetLegacy();
    PrepareLaunchedState(legacy);
    EXPECT_EQ(HCCL_E_RUNTIME, legacy->DeInit());
}

TEST_F(TestCpuThread, Ut_CpuThread_MiscAccessors_When_Normal_Expect_DefaultValues)
{
    CpuThread thread;
    EXPECT_EQ(nullptr, thread.GetStreamLitePtr());
    EXPECT_EQ(false, thread.GetMaster());
    thread.SetIsMaster(true);
    EXPECT_TRUE(thread.GetMaster());
    // dpu 流是进程内资源，不参与跨进程序列化，uniqueId 保持为空
    EXPECT_EQ("", thread.GetUniqueId());
    EXPECT_NO_THROW(thread.LaunchTask());
    EXPECT_NO_THROW(thread.TryLaunchTask());
}
