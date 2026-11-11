/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 新方案控制面的用例：SQ/CQ 布局契约、任务管道的进出与两重退出、队列模式与旧方案的互斥、
// CQ 描述登记与地址投递 kernel、以及 device 侧 CQ 轮询的读表/上报链
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>
#include "../../../hccl_api_base_test.h"
#include "cpu_thread.h"
#include "dfx/res_pub.h"
#include "securec.h"
#include "dpu_sync_layout.h"
#include "cpu_device_sq_submitter.h"
#include "cpu_task_pipe.h"
#include "flush_manager.h"
#include "cpu_device_queue_namecard_restore_kernel.h"
#include "cpu_device_thread.h"
#include "cpu_device_thread_registry.h"
#include "dpu_task_exception_lite.h"
#include "coll_comm_aicpu.h"
#include "hcclCommDfxLite.h"
#include "hcclCommTaskExceptionLite.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "driver/ascend_hal.h"
#include "cpu_thread_test_mock_deps.h"

using namespace hccl;
// Hrt* 系列在 namespace Hccl（extern "C" 链接），用目标 using 声明引入，
// 避免 `using namespace Hccl;` 与 hccl 命名空间产生重名二义
using Hccl::HrtGetDevicePhyIdByUserDevId;
using Hccl::HrtHalGetDeviceInfo;
using Hccl::HrtHalHostRegister;
using Hccl::HrtHalHostUnregister;
using Hccl::HrtResetXpuDevice;
using Hccl::HrtSetXpuDevice;

namespace {
constexpr uint8_t EXIT_SIGNAL = 3; // 旧方案 TaskService 通道的退出回执（DEVICE_SIGNAL_THIRD）

// 上游 DPU 用例仅需要固定测试容量，非资源上限。
constexpr uint32_t NOTIFY_TEST_NUM = 256U;

// 与真实 DPUTAG 同形的两块假半区：半区内的偏移完全照 DPU_*_OFF 摆，只是把 50MB 缩到刚够用，
// 让「名片算得对不对」这件事在没有 100MB 申请的前提下也能验
constexpr uint64_t FAKE_HALF_BYTES = DPU_SQ_ELEM_OFF + DPU_SQ_ELEM_BYTES;
// 两半区共用这个尺寸：CQ 侧再长就越过 vector 尾部，Poke/Peek 会变成堆越界
static_assert(DPU_CQ_ELEM_OFF + DPU_CQ_ELEM_BYTES <= FAKE_HALF_BYTES, "CQ 半区超出 FakeDputag 缩小缓冲");

class FakeDputag {
public:
    FakeDputag() : buf_(2 * FAKE_HALF_BYTES, 0xAA) // 全填垃圾：Init 若不清零，用例就会看到 0xAA
    {
        const uint64_t sqHalf = reinterpret_cast<uint64_t>(buf_.data());
        const uint64_t cqHalf = sqHalf + FAKE_HALF_BYTES;
        layout_.sqMeta = sqHalf + DPU_SQ_META_OFF;
        layout_.sqElem = sqHalf + DPU_SQ_ELEM_OFF;
        layout_.cqMeta = cqHalf + DPU_CQ_META_OFF;
        layout_.cqElem = cqHalf + DPU_CQ_ELEM_OFF;
        layout_.sqDepth = DPU_SQ_DEPTH;
        layout_.cqDepth = DPU_CQ_DEPTH;
        layout_.elemSize = DPU_SQE_SIZE;
    }

    const DpuSyncLayout& Layout() const { return layout_; }

    // host视角（本轮 kernel 与host同址）的 CQ 描述，供守护用例喂 PollOneComm
    QueueDescriptor CqDesc() const
    {
        QueueDescriptor desc{};
        desc.base = layout_.cqElem;
        desc.depth = layout_.cqDepth;
        desc.elemSize = DPU_CQE_SIZE;
        desc.piAddr = layout_.cqMeta + offsetof(DpuCqMeta, pi);
        desc.ciAddr = layout_.cqMeta + offsetof(DpuCqMeta, ci);
        return desc;
    }

    // host视角的 SQ 描述，与 cpu_thread.cc 的 GetDevSyncLayout 同形
    QueueDescriptor SqDesc() const
    {
        QueueDescriptor desc{};
        desc.base = layout_.sqElem;
        desc.depth = layout_.sqDepth;
        desc.elemSize = DPU_SQE_SIZE;
        desc.piAddr = layout_.sqMeta + offsetof(DpuSqMeta, pi);
        desc.ciAddr = layout_.sqMeta + offsetof(DpuSqMeta, ci);
        return desc;
    }

    DpuSqe PeekSqElem(uint32_t slot) const
    {
        DpuSqe sqe{};
        EXPECT_EQ(
            EOK, memcpy_s(
                     &sqe, sizeof(sqe), reinterpret_cast<const void*>(layout_.sqElem + slot * layout_.elemSize),
                     sizeof(sqe)));
        return sqe;
    }

    void PokeSqElem(uint32_t slot, const DpuSqe& sqe)
    {
        ASSERT_EQ(
            EOK,
            memcpy_s(
                reinterpret_cast<void*>(layout_.sqElem + slot * layout_.elemSize), sizeof(sqe), &sqe, sizeof(sqe)));
    }

    void PokeSqPi(uint32_t pi) { ASSERT_EQ(EOK, memcpy_s(&pipeSqMeta()->pi, sizeof(pi), &pi, sizeof(pi))); }

    void PokeSqCi(uint32_t ci) { ASSERT_EQ(EOK, memcpy_s(&pipeSqMeta()->ci, sizeof(ci), &ci, sizeof(ci))); }

    void PokeCq(uint32_t pi, uint32_t ci)
    {
        ASSERT_EQ(EOK, memcpy_s(&pipeCqMeta()->pi, sizeof(pi), &pi, sizeof(pi)));
        ASSERT_EQ(EOK, memcpy_s(&pipeCqMeta()->ci, sizeof(ci), &ci, sizeof(ci)));
    }

    void PokeCqElem(uint32_t slot, const DpuCqe& cqe)
    {
        ASSERT_EQ(
            EOK,
            memcpy_s(
                reinterpret_cast<void*>(layout_.cqElem + slot * layout_.elemSize), sizeof(cqe), &cqe, sizeof(cqe)));
    }

    DpuCqe PeekCqElem(uint32_t slot) const
    {
        DpuCqe cqe{};
        EXPECT_EQ(
            EOK, memcpy_s(
                     &cqe, sizeof(cqe), reinterpret_cast<const void*>(layout_.cqElem + slot * layout_.elemSize),
                     sizeof(cqe)));
        return cqe;
    }

    DpuSqMeta* pipeSqMeta() const { return reinterpret_cast<DpuSqMeta*>(layout_.sqMeta); }
    DpuCqMeta* pipeCqMeta() const { return reinterpret_cast<DpuCqMeta*>(layout_.cqMeta); }

private:
    std::vector<uint8_t> buf_;
    DpuSyncLayout layout_{};
};

// RECORD_TO_NPU 执行分支的 halResourceConfig 入参捕获：SQE notifyId 到 HAL 入参的映射只有这一处
struct HalResourceConfigArgs {
    uint32_t devId;
    drvIdType_t type;
    uint32_t tsId;
    uint32_t resourceId;
    drvResourceConfigType_t prop;
    uint32_t value;
} g_halResourceConfigArgs;

drvError_t StubHalResourceConfig(uint32_t devId, struct halResourceIdInputInfo* in, struct halResourceConfigInfo* para)
{
    g_halResourceConfigArgs.devId = devId;
    g_halResourceConfigArgs.type = in->type;
    g_halResourceConfigArgs.tsId = in->tsId;
    g_halResourceConfigArgs.resourceId = in->resourceId;
    g_halResourceConfigArgs.prop = para->prop;
    g_halResourceConfigArgs.value = para->value[0];
    return DRV_ERROR_NONE;
}

int g_memcpyCallCount = 0;
int g_failAtCall = 0;

// HandleDpuCq 里 memcpy_s 失败分支的注入桩：第 g_failAtCall 次调用报失败、其余按等价语义
// 转发 libc 拷贝（被 mock 的符号自身不可直调）
errno_t StubMemcpyFailAt(void* dest, size_t destMax, const void* src, size_t count)
{
    (void)destMax;
    ++g_memcpyCallCount;
    if (g_memcpyCallCount == g_failAtCall) {
        return EINVAL;
    }
    (void)std::memcpy(dest, src, count);
    return EOK;
}

// 队列模式拉起：身份注入与 1 thread 1 kernel 的交叉判据都在 CpuThread 上，故走 thread.InitQueueMode
HcclResult InitQueueMode(CpuThread& thread)
{
    Hccl::TaskServiceArgs args{"testCommId", 0};
    return thread.InitQueueMode(args);
}

// 模拟 kernel：真实 kernel 下发到 dpu 流上跑，UT 里没有流，改本地线程跑 pipe->Run()。
// 队列通道的终止靠 destroyFlag_（host 置 true、kernel 写回 false）：不起这个消费者，
// 后续 DeInit 的 WaitKernelTerminate 就永远等不到回写
void StartKernelSim(CpuThread& thread, std::thread& sim)
{
    sim = std::thread([pipe = thread.GetTaskPipe()]() {
        (void)pipe->Run();
    });
}

// 有界等待：用例必须自己保证「等不到也中止」，否则失败会挂死整个 UT
bool WaitUntil(const std::function<bool()>& pred, uint32_t timeoutMs = 2000)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}

// 游标在生产侧走 __atomic_load_n/__atomic_store_n，这里同口径读：
// 一边原子一边普通读且并发，就是数据竞争
uint32_t LoadCursor(const uint32_t* cursor) { return __atomic_load_n(cursor, __ATOMIC_ACQUIRE); }

// 断言失败会让用例中途返回，此时 joinable 的 std::thread 一析构就 std::terminate：
// 守卫先把被测线程停下再 join，让失败只终止这个用例
class ThreadJoiner {
public:
    ThreadJoiner(std::thread& thread, std::function<void()> stop) : thread_(thread), stop_(std::move(stop)) {}

    ~ThreadJoiner()
    {
        if (!thread_.joinable()) {
            return;
        }
        stop_();
        thread_.join();
    }

    ThreadJoiner(const ThreadJoiner&) = delete;
    ThreadJoiner& operator=(const ThreadJoiner&) = delete;

private:
    std::thread& thread_;
    std::function<void()> stop_;
};

constexpr uint32_t PEER_NOTIFY_ID = 9u;

// channel 级失败分支的公共现场：桩已由用例挂好，一条 SQE 喂给 Run，
// 断言以注入码退出、ci 停在出错格、CQE 按 hcclErrCode/taskId/streamId 回执
void RunSqeExpectErrorCqe(FakeDputag& mem, CpuTaskPipe& pipe, const DpuSqe& sqe, HcclResult expected)
{
    pipe.SetDfxStreamId(6);
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::atomic<int32_t> runRet{0};
    std::thread runner([&pipe, &exited, &runRet]() {
        runRet = static_cast<int32_t>(pipe.Run());
        exited = true;
    });
    ASSERT_TRUE(WaitUntil([&exited]() {
        return exited.load();
    }));
    runner.join();

    EXPECT_EQ(static_cast<int32_t>(expected), runRet.load());
    EXPECT_EQ(0u, pipe.SqMeta()->ci); // 失败格不消费：SQ 停在出错现场
    EXPECT_EQ(1u, pipe.CqMeta()->pi); // 失败须回执
    const DpuCqe cqe = mem.PeekCqElem(0);
    EXPECT_EQ(static_cast<int32_t>(expected), cqe.hcclErrCode);
    EXPECT_EQ(sqe.taskId, cqe.taskId);
    EXPECT_EQ(6u, cqe.streamId); // dfxStreamId_ 随回执带出，镜像队列按它认领
}

// TO_NPU 退化臂的对端 NPU 线程：notify 地址口径不适用（GetNotifyHostShmAddr 走基类默认
// NOT_SUPPORT），notifyId 按固定值给出，供 TO_NPU SQE 的 notifyId 字段断言
class NpuPeerThread : public Thread {
public:
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
    HcclResult GetNotifyId(uint32_t notifyIdx, uint32_t& notifyId) const override
    {
        notifyId = PEER_NOTIFY_ID;
        return HCCL_SUCCESS;
    }

private:
    std::string uniqueId_{"npu_peer_thread"};
};
} // namespace

class TestCpuTaskPipe : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // main.cc 预置了非法的 HCCL_DFS_CONFIG=connection_fault_detction_time:0，
        // LaunchDpuKernelWithArgs 经 EnvConfig 读到会抛异常，须先清除（同 test_cpu_thread_class.cc）
        unsetenv("HCCL_DFS_CONFIG");
        CpuThreadBase::commNum_ = 0;
        ClearCpuDeviceThread();
    }
    void TearDown() override
    {
        ClearCpuDeviceThread();
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

// ==================== 布局契约 ====================

TEST_F(TestCpuTaskPipe, Ut_Layout_When_StaticContract_Expect_ExactSizes)
{
    // 头里已有 static_assert，这里运行期再钉一遍：跨核 POD 的尺寸一旦漂移就是静默错位
    EXPECT_EQ(64u, sizeof(DpuSqe));
    EXPECT_EQ(24u, offsetof(DpuSqe, u));
    EXPECT_EQ(24u, offsetof(DpuSqe, u.channelWrite.channelHandle)); // channelHandle 已下沉进 channel 变体首字段
    EXPECT_EQ(64u, sizeof(DpuCqe));
    EXPECT_EQ(4u, offsetof(DpuCqe, taskId));
    EXPECT_EQ(8u, offsetof(DpuCqe, streamId));
    EXPECT_EQ(16u, offsetof(DpuCqe, extra));
    EXPECT_EQ(4096u, sizeof(DpuSqMeta));
    EXPECT_EQ(4096u, sizeof(DpuCqMeta));
    EXPECT_EQ(16u, offsetof(DpuCqMeta, opIndex));
    EXPECT_EQ(32u, sizeof(QueueDescriptor));
    EXPECT_EQ(48u, sizeof(DpuSyncLayout));
    EXPECT_EQ(16u, sizeof(DpuQueueLaunchParam));
    EXPECT_EQ(8u, offsetof(DpuQueueLaunchParam, deviceId));
    EXPECT_EQ(0u, offsetof(DpuQueueNamecardParam, cqDesc));
    EXPECT_EQ(1024u, DPU_SQ_DEPTH);
    EXPECT_EQ(1u, DPU_CQ_DEPTH);
    EXPECT_EQ(1023u, DPU_SQ_USABLE_DEPTH);
    EXPECT_EQ(128ULL * 1024, DPU_TAG_HALF_BYTES);
    // 两半区都装得下各自的队列（静态断言的运行期对照）
    EXPECT_LE(DPU_SQ_ELEM_OFF + DPU_SQ_ELEM_BYTES, DPU_TAG_HALF_BYTES);
    EXPECT_LE(DPU_CQ_ELEM_OFF + DPU_CQ_ELEM_BYTES, DPU_TAG_HALF_BYTES);
    // SQ 元数据落半区偏移 0：终止/复位握手已不落共享内存，无需再留位
    EXPECT_EQ(0ULL, DPU_SQ_META_OFF);
    EXPECT_EQ(0ULL, DPU_CQ_META_OFF);
}

// ==================== CpuTaskPipe：建队列 ====================

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Init_When_LayoutGiven_Expect_MetaZeroedAndFilled)
{
    FakeDputag mem;
    CpuTaskPipe pipe;

    EXPECT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));
    EXPECT_TRUE(pipe.IsInited());
    // DpuShmemMgr::Alloc 只清首字节，游标与算子序号必须是本步显式清零的（假半区预填 0xAA）
    EXPECT_EQ(DPU_SQ_DEPTH, pipe.SqMeta()->depth);
    EXPECT_EQ(DPU_SQE_SIZE, pipe.SqMeta()->elemSize);
    EXPECT_EQ(0u, pipe.SqMeta()->pi);
    EXPECT_EQ(0u, pipe.SqMeta()->ci);
    EXPECT_EQ(DPU_CQ_DEPTH, pipe.CqMeta()->depth);
    EXPECT_EQ(DPU_SQE_SIZE, pipe.CqMeta()->elemSize);
    EXPECT_EQ(0u, pipe.CqMeta()->pi);
    EXPECT_EQ(0u, pipe.CqMeta()->ci);
    EXPECT_EQ(0u, pipe.CqMeta()->opIndex);

    // 幂等：二次 Init 无副作用（游标仍保持清零）
    EXPECT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));
    EXPECT_TRUE(pipe.IsInited());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Init_When_LayoutInvalid_Expect_Para)
{
    CpuTaskPipe pipe;
    FakeDputag mem;
    DpuSyncLayout layout = mem.Layout();

    DpuSyncLayout nullLayout{};
    EXPECT_EQ(HCCL_E_PARA, pipe.Init(nullLayout));
    EXPECT_FALSE(pipe.IsInited());

    layout.sqDepth = 100U; // SQ 非 2 的幂
    EXPECT_EQ(HCCL_E_PARA, pipe.Init(layout));

    layout = mem.Layout();
    layout.sqDepth = 0U;
    EXPECT_EQ(HCCL_E_PARA, pipe.Init(layout));

    layout = mem.Layout();
    layout.cqDepth = 0U; // CQ 槽数不可为 0
    EXPECT_EQ(HCCL_E_PARA, pipe.Init(layout));

    layout = mem.Layout();
    layout.elemSize = DPU_SQE_SIZE + 1U; // 格宽必须是 SQE 宽
    EXPECT_EQ(HCCL_E_PARA, pipe.Init(layout));

    EXPECT_FALSE(pipe.IsInited());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_NotInited_Expect_Internal)
{
    CpuTaskPipe pipe;
    EXPECT_EQ(HCCL_E_INTERNAL, pipe.Run());
}

// ==================== CpuTaskPipe：destroy/reset 退出 ====================

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_DestroyFlag_Expect_ExitAndAck)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    pipe.RequestDestroy();               // host 写 destroyFlag_
    EXPECT_EQ(HCCL_SUCCESS, pipe.Run()); // Run 读到即写回并退出
    // IsDestroyAcked 还认 runExited_，只有直接读标志才验得到「kernel 把 destroyFlag_ 写回 false」
    EXPECT_FALSE(pipe.destroyFlag_.load(std::memory_order_acquire));
    EXPECT_TRUE(pipe.IsDestroyAcked());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ErrorExit_Then_Destroy_Expect_Acked)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    DpuSqe sqe{};
    sqe.opKind = 0x18; // 不支持的类型：Run 直接返回，不会写回 destroyFlag_
    sqe.taskId = 7;
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, pipe.Run());

    pipe.RequestDestroy(); // kernel 已退出后才来终止，仍应立刻拿到回执
    EXPECT_TRUE(pipe.IsDestroyAcked());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ResetFlag_Expect_ClearSqAndResetCursorWithoutExit)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    // 先让消费侧取走一条把 ci 推到 1，再补一条抬高 pi：reset 应把两者归零、清空元素区且不退出
    uint8_t scratch[8] = {0};
    pipe.SetNotifyInfo(sizeof(scratch), scratch);
    DpuSqe sqe{};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_RECORD_TO_DPU;
    sqe.taskId = 7;
    sqe.u.threadNotifyRecordToDpu.addr = reinterpret_cast<uint64_t>(scratch);
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::thread runner([&pipe, &exited]() {
        (void)pipe.Run();
        exited = true;
    });

    EXPECT_TRUE(WaitUntil([&pipe]() {
        return LoadCursor(&pipe.SqMeta()->ci) == 1u;
    }));

    mem.PokeSqElem(1, sqe);
    mem.PokeSqPi(2);
    pipe.RequestReset();

    EXPECT_TRUE(WaitUntil([&pipe]() {
        return LoadCursor(&pipe.SqMeta()->pi) == 0u && LoadCursor(&pipe.SqMeta()->ci) == 0u;
    }));
    EXPECT_FALSE(exited.load());             // reset 只清空、不退出轮询
    EXPECT_EQ(0u, mem.PeekSqElem(0).taskId); // 元素区已清零

    pipe.RequestDestroy(); // 收尾：让 runner 退出
    runner.join();
    EXPECT_TRUE(exited.load());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_UnsupportedSqe_Expect_StopWithNotSupportCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));
    pipe.SetDfxStreamId(6);

    DpuSqe sqe{};
    sqe.opKind = 0x18;
    sqe.taskId = 7;
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::atomic<int32_t> runRet{0};
    std::thread runner([&pipe, &exited, &runRet]() {
        runRet = static_cast<int32_t>(pipe.Run());
        exited = true;
    });
    ThreadJoiner joiner(runner, [&pipe]() {
        pipe.RequestDestroy();
    });

    ASSERT_TRUE(WaitUntil([&exited]() {
        return exited.load();
    }));
    runner.join();
    EXPECT_EQ(static_cast<int32_t>(HCCL_E_NOT_SUPPORT), runRet.load());
    EXPECT_EQ(0u, pipe.SqMeta()->ci); // 失败格不消费：SQ 停在出错现场
    EXPECT_EQ(1u, pipe.CqMeta()->pi); // 不支持的 opKind 同样要回执
    const DpuCqe cqe = mem.PeekCqElem(0);
    EXPECT_EQ(static_cast<int32_t>(HCCL_E_NOT_SUPPORT), cqe.hcclErrCode);
    EXPECT_EQ(7u, cqe.taskId);
    EXPECT_EQ(6u, cqe.streamId);
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_LocalCopySqe_Expect_DstCopiedAndCiAdvances)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    uint8_t srcBuf[] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t dstBuf[sizeof(srcBuf)] = {0};
    DpuSqe sqe{};
    sqe.opKind = DPU_OP_LOCAL_COPY;
    sqe.taskId = 7;
    sqe.u.localCopy.dst = reinterpret_cast<uint64_t>(dstBuf);
    sqe.u.localCopy.src = reinterpret_cast<uint64_t>(srcBuf);
    sqe.u.localCopy.len = sizeof(srcBuf);
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::thread runner([&pipe, &exited]() {
        (void)pipe.Run();
        exited = true;
    });

    bool picked = WaitUntil([&pipe]() {
        return LoadCursor(&pipe.SqMeta()->ci) == 1u;
    });
    EXPECT_TRUE(picked);
    EXPECT_EQ(0, memcmp(dstBuf, srcBuf, sizeof(srcBuf)));

    pipe.RequestDestroy();
    runner.join();
    EXPECT_TRUE(exited.load());
    EXPECT_EQ(1u, pipe.SqMeta()->ci);
}

// ==================== CpuTaskPipe：Run 侧 thread 级 notify 分派 ====================

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ThreadNotifyWaitSqe_Expect_ConsumeAndZeroBack)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    uint8_t notifyShmem[NOTIFY_TEST_NUM] = {0};
    pipe.SetNotifyInfo(NOTIFY_TEST_NUM, notifyShmem);
    notifyShmem[3] = 1; // 对端已 record：wait 首轮即命中

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_WAIT;
    sqe.taskId = 7;
    sqe.u.threadNotifyWait.notifyIdx = 3;
    sqe.u.threadNotifyWait.timeOut = 2;
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::thread runner([&pipe, &exited]() {
        (void)pipe.Run();
        exited = true;
    });
    ThreadJoiner joiner(runner, [&pipe]() {
        pipe.RequestDestroy();
    });

    ASSERT_TRUE(WaitUntil([&pipe]() {
        return LoadCursor(&pipe.SqMeta()->ci) == 1u;
    }));
    EXPECT_EQ(0u, notifyShmem[3]); // 消费即清零：同一个字节供下一轮 record/wait 复用
    EXPECT_FALSE(exited.load());   // wait 命中不退出，继续等host驱动

    pipe.RequestDestroy();
    runner.join();
    EXPECT_TRUE(exited.load());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ThreadNotifyWaitTimeout_Expect_TimeoutCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    uint8_t notifyShmem[NOTIFY_TEST_NUM] = {0}; // 恒为 0：等不到对端 record
    pipe.SetNotifyInfo(NOTIFY_TEST_NUM, notifyShmem);
    pipe.SetDfxStreamId(6);

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_WAIT;
    sqe.taskId = 7;
    sqe.u.threadNotifyWait.notifyIdx = 3;
    sqe.u.threadNotifyWait.timeOut = 0; // 0 秒预算：首轮退避后即超时，不必真等满秒
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::atomic<int32_t> runRet{0};
    std::thread runner([&pipe, &exited, &runRet]() {
        runRet = static_cast<int32_t>(pipe.Run());
        exited = true;
    });
    ThreadJoiner joiner(runner, [&pipe]() {
        pipe.RequestDestroy();
    });
    ASSERT_TRUE(WaitUntil([&exited]() {
        return exited.load();
    }));
    runner.join();

    EXPECT_EQ(static_cast<int32_t>(HCCL_E_TIMEOUT), runRet.load());
    EXPECT_EQ(0u, pipe.SqMeta()->ci); // 失败格不消费：SQ 停在出错现场
    EXPECT_EQ(1u, pipe.CqMeta()->pi); // 失败须回执
    const DpuCqe cqe = mem.PeekCqElem(0);
    EXPECT_EQ(static_cast<int32_t>(HCCL_E_TIMEOUT), cqe.hcclErrCode);
    EXPECT_EQ(7u, cqe.taskId);
    EXPECT_EQ(6u, cqe.streamId); // dfxStreamId_ 随回执带出，镜像队列按它认领
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ThreadNotifyWaitNoShmem_Expect_InternalCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));
    // 不 SetNotifyInfo：通知内存未接线是配置错误，必须立即失败而不是空转轮询

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_WAIT;
    sqe.taskId = 9;
    sqe.u.threadNotifyWait.notifyIdx = 0;
    sqe.u.threadNotifyWait.timeOut = 2;
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::atomic<int32_t> runRet{0};
    std::thread runner([&pipe, &exited, &runRet]() {
        runRet = static_cast<int32_t>(pipe.Run());
        exited = true;
    });
    ThreadJoiner joiner(runner, [&pipe]() {
        pipe.RequestDestroy();
    });
    ASSERT_TRUE(WaitUntil([&exited]() {
        return exited.load();
    }));
    runner.join();

    EXPECT_EQ(static_cast<int32_t>(HCCL_E_INTERNAL), runRet.load());
    EXPECT_EQ(0u, pipe.SqMeta()->ci);
    EXPECT_EQ(1u, pipe.CqMeta()->pi);
    const DpuCqe cqe = mem.PeekCqElem(0);
    EXPECT_EQ(static_cast<int32_t>(HCCL_E_INTERNAL), cqe.hcclErrCode);
    EXPECT_EQ(9u, cqe.taskId);
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ThreadNotifyRecordToDpuSqe_Expect_NotifyByteMarked)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    uint8_t notifyShmem[NOTIFY_TEST_NUM] = {0};
    pipe.SetNotifyInfo(NOTIFY_TEST_NUM, notifyShmem);
    DpuSqe sqe{};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_RECORD_TO_DPU;
    sqe.taskId = 7;
    // SQE 烧 host 视角绝对地址（执行段在 host 侧直写），对端 wait 轮询的就是这个字节
    sqe.u.threadNotifyRecordToDpu.addr = reinterpret_cast<uint64_t>(notifyShmem + 5);
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::thread runner([&pipe, &exited]() {
        (void)pipe.Run();
        exited = true;
    });
    ThreadJoiner joiner(runner, [&pipe]() {
        pipe.RequestDestroy();
    });

    ASSERT_TRUE(WaitUntil([&pipe]() {
        return LoadCursor(&pipe.SqMeta()->ci) == 1u;
    }));
    EXPECT_EQ(1u, notifyShmem[5]);
    EXPECT_FALSE(exited.load());

    pipe.RequestDestroy();
    runner.join();
    EXPECT_TRUE(exited.load());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ThreadNotifyRecordToNpuSqe_Expect_RecordToNotifyId)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));
    pipe.SetDevId(5);

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_RECORD_TO_NPU;
    sqe.taskId = 7;
    sqe.u.threadNotifyRecordToNpu.notifyId = 9;
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::thread runner([&pipe, &exited]() {
        (void)pipe.Run();
        exited = true;
    });
    ThreadJoiner joiner(runner, [&pipe]() {
        pipe.RequestDestroy();
    });

    // 第一条走桩库真身：halResourceConfig 弱桩返回 DRV_ERROR_NONE 即本分支的成功路径
    ASSERT_TRUE(WaitUntil([&pipe]() {
        return LoadCursor(&pipe.SqMeta()->ci) == 1u;
    }));
    EXPECT_FALSE(exited.load());

    // 第二条换捕获桩：SQE notifyId 到 HAL 入参的映射只在此处可验
    MOCKER(halResourceConfig).stubs().will(invoke(StubHalResourceConfig));
    mem.PokeSqElem(1, sqe);
    mem.PokeSqPi(2);
    ASSERT_TRUE(WaitUntil([&pipe]() {
        return LoadCursor(&pipe.SqMeta()->ci) == 2u;
    }));
    EXPECT_EQ(5u, g_halResourceConfigArgs.devId);
    EXPECT_EQ(DRV_NOTIFY_ID, g_halResourceConfigArgs.type);
    EXPECT_EQ(0u, g_halResourceConfigArgs.tsId);
    EXPECT_EQ(9u, g_halResourceConfigArgs.resourceId);
    EXPECT_EQ(DRV_ID_RECORD, g_halResourceConfigArgs.prop);
    EXPECT_EQ(1u, g_halResourceConfigArgs.value);

    pipe.RequestDestroy();
    runner.join();
    EXPECT_TRUE(exited.load());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ThreadNotifyRecordToNpuFailed_Expect_InternalCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));
    pipe.SetDevId(5);

    MOCKER(halResourceConfig).stubs().will(returnValue(DRV_ERROR_INNER_ERR));

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_RECORD_TO_NPU;
    sqe.taskId = 7;
    sqe.u.threadNotifyRecordToNpu.notifyId = 9;
    mem.PokeSqElem(0, sqe);
    mem.PokeSqPi(1);

    std::atomic<bool> exited{false};
    std::atomic<int32_t> runRet{0};
    std::thread runner([&pipe, &exited, &runRet]() {
        runRet = static_cast<int32_t>(pipe.Run());
        exited = true;
    });
    ThreadJoiner joiner(runner, [&pipe]() {
        pipe.RequestDestroy();
    });
    ASSERT_TRUE(WaitUntil([&exited]() {
        return exited.load();
    }));
    runner.join();

    EXPECT_EQ(static_cast<int32_t>(HCCL_E_INTERNAL), runRet.load()); // HAL 失败必须转成回执而非静默
    EXPECT_EQ(0u, pipe.SqMeta()->ci);
    EXPECT_EQ(1u, pipe.CqMeta()->pi);
    const DpuCqe cqe = mem.PeekCqElem(0);
    EXPECT_EQ(static_cast<int32_t>(HCCL_E_INTERNAL), cqe.hcclErrCode);
    EXPECT_EQ(7u, cqe.taskId);
}

// ==================== CpuTaskPipe：Run 侧 channel 级分派 ====================

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ChannelOpsSqe_Expect_ConsumedWithoutCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    // 原语真身按句柄解引用 channel 实体，UT 无实体只能全桩死；
    // 各桩 expects(once())：分派错走旁支原语时由 TearDown 的 verify 兜底挂出
    MOCKER(HcommChannelNotifyWaitOnThread).expects(once()).will(returnValue(static_cast<int32_t>(HCCL_SUCCESS)));
    MOCKER(HcommChannelNotifyRecordOnThread).expects(once()).will(returnValue(static_cast<int32_t>(HCCL_SUCCESS)));
    MOCKER(HcommWriteWithNotifyNbiOnThread).expects(once()).will(returnValue(static_cast<int32_t>(HCCL_SUCCESS)));
    MOCKER(HcommChannelDrainOnThread).expects(once()).will(returnValue(static_cast<int32_t>(HCCL_SUCCESS)));
    MOCKER(HcommChannelFenceOnThread).expects(once()).will(returnValue(static_cast<int32_t>(HCCL_SUCCESS)));
    MOCKER(&Hccl::FlushManager::Flush).expects(once()).will(returnValue(HCCL_SUCCESS));

    // 6 个 channel 级分支一次喂齐：channelHandle 已下沉为各变体首字段（union 同偏移）
    const ChannelHandle channel = 0x3000ULL;
    auto pokeChannelSqe = [&mem, channel](uint32_t slot, DpuOpKind opKind, uint32_t taskId) {
        DpuSqe sqe{};
        sqe.opKind = opKind;
        sqe.taskId = taskId;
        sqe.u.channelNotifyWait.channelHandle = channel;
        mem.PokeSqElem(slot, sqe);
    };
    pokeChannelSqe(0, DPU_OP_CHANNEL_NOTIFY_WAIT, 11);
    pokeChannelSqe(1, DPU_OP_CHANNEL_NOTIFY_RECORD, 12);
    pokeChannelSqe(2, DPU_OP_CHANNEL_WRITE_WITH_NOTIFY_NBI, 13);
    pokeChannelSqe(3, DPU_OP_CHANNEL_DRAIN, 14);
    pokeChannelSqe(4, DPU_OP_CHANNEL_FENCE, 15);
    pokeChannelSqe(5, DPU_OP_COMM_FENCE, 16);
    mem.PokeSqPi(6);

    std::atomic<bool> exited{false};
    std::thread runner([&pipe, &exited]() {
        (void)pipe.Run();
        exited = true;
    });

    ASSERT_TRUE(WaitUntil([&pipe]() {
        return pipe.SqMeta()->ci == 6u;
    }));
    EXPECT_EQ(0u, pipe.CqMeta()->pi); // 成功路径不产生 CQE
    EXPECT_FALSE(exited.load());      // 全部消费后不退出，继续等host驱动

    pipe.RequestDestroy();
    runner.join();
    EXPECT_TRUE(exited.load());
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ChannelNotifyWaitFailed_Expect_InternalCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    MOCKER(HcommChannelNotifyWaitOnThread).stubs().will(returnValue(static_cast<int32_t>(HCCL_E_INTERNAL)));

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_CHANNEL_NOTIFY_WAIT;
    sqe.taskId = 11;
    sqe.u.channelNotifyWait.channelHandle = 0x3000ULL;
    sqe.u.channelNotifyWait.notifyIdx = 5;
    sqe.u.channelNotifyWait.timeOut = 9;
    RunSqeExpectErrorCqe(mem, pipe, sqe, HCCL_E_INTERNAL);
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ChannelNotifyRecordFailed_Expect_InternalCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    MOCKER(HcommChannelNotifyRecordOnThread).stubs().will(returnValue(static_cast<int32_t>(HCCL_E_INTERNAL)));

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_CHANNEL_NOTIFY_RECORD;
    sqe.taskId = 12;
    sqe.u.channelNotifyRecord.channelHandle = 0x3000ULL;
    sqe.u.channelNotifyRecord.notifyIdx = 5;
    RunSqeExpectErrorCqe(mem, pipe, sqe, HCCL_E_INTERNAL);
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ChannelWriteWithNotifyNbiFailed_Expect_InternalCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    MOCKER(HcommWriteWithNotifyNbiOnThread).stubs().will(returnValue(static_cast<int32_t>(HCCL_E_INTERNAL)));

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_CHANNEL_WRITE_WITH_NOTIFY_NBI;
    sqe.taskId = 13;
    sqe.u.channelWriteWithNotify.channelHandle = 0x3000ULL;
    sqe.u.channelWriteWithNotify.dstAddr = 0x1000ULL;
    sqe.u.channelWriteWithNotify.srcAddr = 0x2000ULL;
    sqe.u.channelWriteWithNotify.len = 64;
    sqe.u.channelWriteWithNotify.notifyIdx = 5;
    RunSqeExpectErrorCqe(mem, pipe, sqe, HCCL_E_INTERNAL);
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ChannelDrainFailed_Expect_InternalCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    MOCKER(HcommChannelDrainOnThread).stubs().will(returnValue(static_cast<int32_t>(HCCL_E_INTERNAL)));

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_CHANNEL_DRAIN;
    sqe.taskId = 14;
    sqe.u.channelDrainOrFence.channelHandle = 0x3000ULL;
    RunSqeExpectErrorCqe(mem, pipe, sqe, HCCL_E_INTERNAL);
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_ChannelFenceFailed_Expect_InternalCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    MOCKER(HcommChannelFenceOnThread).stubs().will(returnValue(static_cast<int32_t>(HCCL_E_INTERNAL)));

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_CHANNEL_FENCE;
    sqe.taskId = 15;
    sqe.u.channelDrainOrFence.channelHandle = 0x3000ULL;
    RunSqeExpectErrorCqe(mem, pipe, sqe, HCCL_E_INTERNAL);
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_Run_When_CommFenceFailed_Expect_InternalCqe)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    MOCKER(&Hccl::FlushManager::Flush).stubs().will(returnValue(HCCL_E_INTERNAL));

    DpuSqe sqe{};
    sqe.opKind = DPU_OP_COMM_FENCE;
    sqe.taskId = 16;
    RunSqeExpectErrorCqe(mem, pipe, sqe, HCCL_E_INTERNAL);
}

// ==================== CpuDeviceSqSubmitter：SQ 生产侧（数据面填 SQE，本接口入队） ====================

// 每个用例先 pipe.Init：它清零两页元数据，这正是host在投递名片之前的动作，
// 之后 submitter.Init 读到的初始 pi 才是 0 而不是假半区的 0xAA
TEST_F(TestCpuTaskPipe, Ut_SqSubmitter_Init_When_DescInvalid_Expect_Para)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    CpuDeviceSqSubmitter submitter;
    EXPECT_FALSE(submitter.IsInited());
    // 未 Init 就提交：给调用方一次明确的失败，而不是往野地址里写
    DpuSqe sqe{};
    uint32_t taskId = 0;
    EXPECT_EQ(HCCL_E_INTERNAL, submitter.Submit(sqe, taskId));

    QueueDescriptor bad{};
    EXPECT_EQ(HCCL_E_PARA, submitter.Init(bad)); // 全 0 名片

    bad = mem.SqDesc();
    bad.elemSize = DPU_CQE_SIZE + 1U; // 元素宽度与布局契约不一致
    EXPECT_EQ(HCCL_E_PARA, submitter.Init(bad));

    bad = mem.SqDesc();
    bad.depth = 1000U; // 非 2 的幂：取模环撑不住
    EXPECT_EQ(HCCL_E_PARA, submitter.Init(bad));

    EXPECT_FALSE(submitter.IsInited());

    ASSERT_EQ(HCCL_SUCCESS, submitter.Init(mem.SqDesc()));
    EXPECT_TRUE(submitter.IsInited());
    EXPECT_EQ(mem.SqDesc().base, submitter.GetSqDesc().base);
}

TEST_F(TestCpuTaskPipe, Ut_SqSubmitter_Submit_When_Normal_Expect_ElemWrittenAndPiAdvanced)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    CpuDeviceSqSubmitter submitter;
    ASSERT_EQ(HCCL_SUCCESS, submitter.Init(mem.SqDesc()));
    EXPECT_EQ(0u, pipe.SqMeta()->pi);

    DpuSqe first{};
    first.opKind = DPU_OP_CHANNEL_WRITE;
    first.taskId = 0xFFFFFFFFU; // 调用方填的一律作废：身份由本接口盖章
    first.u.channelWrite.dstAddr = 0x1234U;
    first.u.channelWrite.len = 0x99U;
    uint32_t taskId = 0xFFFFFFFFU;
    ASSERT_EQ(HCCL_SUCCESS, submitter.Submit(first, taskId));

    EXPECT_EQ(0u, taskId); // 从 0 起连续分配
    EXPECT_EQ(1u, pipe.SqMeta()->pi);
    const DpuSqe got = mem.PeekSqElem(0);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_CHANNEL_WRITE), got.opKind);
    EXPECT_EQ(0u, got.taskId);
    EXPECT_EQ(0x1234U, got.u.channelWrite.dstAddr);
    EXPECT_EQ(0x99U, got.u.channelWrite.len);

    DpuSqe second{};
    second.opKind = DPU_OP_CHANNEL_READ;
    ASSERT_EQ(HCCL_SUCCESS, submitter.Submit(second, taskId));
    EXPECT_EQ(1u, taskId);
    EXPECT_EQ(2u, pipe.SqMeta()->pi);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_CHANNEL_READ), mem.PeekSqElem(1).opKind);
}

TEST_F(TestCpuTaskPipe, Ut_SqSubmitter_Submit_When_RingFull_Expect_Timeout)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));
    // 满判据牺牲一格：在途 depth-1 条即视为满，此时 pi 已到 ci + depth - 1
    mem.PokeSqCi(0);
    mem.PokeSqPi(DPU_SQ_USABLE_DEPTH);

    CpuDeviceSqSubmitter submitter;
    ASSERT_EQ(HCCL_SUCCESS, submitter.Init(mem.SqDesc()));

    DpuSqe sqe{};
    uint32_t taskId = 0xFFFFFFFFU;
    EXPECT_EQ(HCCL_E_TIMEOUT, submitter.Submit(sqe, taskId, 0)); // timeOutMs = 0：只探查一次
    EXPECT_EQ(DPU_SQ_USABLE_DEPTH, pipe.SqMeta()->pi);           // 环满不动 pi，也不占 taskId
    EXPECT_EQ(0xFFFFFFFFU, taskId);

    // 对端放走一格后立刻可入队
    mem.PokeSqCi(1);
    ASSERT_EQ(HCCL_SUCCESS, submitter.Submit(sqe, taskId, 0));
    EXPECT_EQ(DPU_SQ_USABLE_DEPTH + 1U, pipe.SqMeta()->pi);
    EXPECT_EQ(0u, taskId);
}

TEST_F(TestCpuTaskPipe, Ut_SqSubmitter_When_ProducerConsumer_Expect_SameSlotIndex)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    CpuDeviceSqSubmitter submitter;
    ASSERT_EQ(HCCL_SUCCESS, submitter.Init(mem.SqDesc()));

    uint8_t scratch[8] = {0};
    pipe.SetNotifyInfo(sizeof(scratch), scratch);
    DpuSqe sqe{};
    sqe.opKind = DPU_OP_THREAD_NOTIFY_RECORD_TO_DPU;
    sqe.u.threadNotifyRecordToDpu.addr = reinterpret_cast<uint64_t>(scratch);
    uint32_t taskId = 0;
    ASSERT_EQ(HCCL_SUCCESS, submitter.Submit(sqe, taskId));
    EXPECT_EQ(0u, taskId);
    // 生产侧按 pi % depth 落格、消费侧按 ci % depth 取格：两侧对同一条任务的定位必须落在同一格
    EXPECT_EQ(1u, pipe.SqMeta()->pi);
    EXPECT_EQ(0u, pipe.SqMeta()->ci);
    EXPECT_EQ(0u, mem.PeekSqElem(0u).taskId);

    std::atomic<bool> exited{false};
    std::thread runner([&pipe, &exited]() {
        (void)pipe.Run();
        exited = true;
    });
    const bool consumed = WaitUntil([&pipe]() {
        return LoadCursor(&pipe.SqMeta()->ci) == 1u;
    });
    EXPECT_TRUE(consumed);
    EXPECT_FALSE(exited.load()); // 取走一条不退出，继续等host驱动
    pipe.RequestDestroy();
    runner.join();
    EXPECT_TRUE(exited.load());
}

// ==================== CpuTaskPipe：回执 ====================

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_PushCqe_When_Normal_Expect_PiAdvanced)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    EXPECT_EQ(HCCL_E_INTERNAL, pipe.PushCqe(DpuCqe{})); // 未建队列
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    DpuCqe cqe{};
    cqe.taskId = 11;
    cqe.streamId = 3;
    cqe.hcclErrCode = static_cast<int32_t>(HCCL_E_INTERNAL);
    EXPECT_EQ(HCCL_SUCCESS, pipe.PushCqe(cqe));
    EXPECT_EQ(1u, pipe.CqMeta()->pi);

    DpuCqe readBack{};
    ASSERT_EQ(
        EOK,
        memcpy_s(&readBack, sizeof(readBack), reinterpret_cast<const void*>(mem.Layout().cqElem), sizeof(readBack)));
    EXPECT_EQ(11u, readBack.taskId);
    EXPECT_EQ(3u, readBack.streamId);
    EXPECT_EQ(static_cast<int32_t>(HCCL_E_INTERNAL), readBack.hcclErrCode);
}

TEST_F(TestCpuTaskPipe, Ut_TaskPipe_PushCqe_When_CqFull_Expect_DropWithoutAdvance)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    mem.PokeCq(DPU_CQ_DEPTH, 0u); // pi - ci == cqDepth 即满（单槽信箱）
    DpuCqe cqe{};
    cqe.taskId = 13;
    EXPECT_EQ(HCCL_SUCCESS, pipe.PushCqe(cqe)); // 上报可以丢
    EXPECT_EQ(DPU_CQ_DEPTH, pipe.CqMeta()->pi);
}

// ==================== 地址投递 kernel ====================

TEST_F(TestCpuTaskPipe, Ut_QueueNamecardRestore_When_ArgsNull_Expect_Fail)
{
    EXPECT_EQ(1u, RunDpuQueueNamecardRestore(nullptr)); // 1 = DPU_QUEUE_NAMECARD_RESTORE_FAIL
}

TEST_F(TestCpuTaskPipe, Ut_QueueNamecardRestore_When_DescIncomplete_Expect_Fail)
{
    FakeDputag mem;
    DpuQueueNamecardParam param{};
    param.cqDesc = mem.CqDesc();
    param.cqDesc.piAddr = 0; // 游标地址缺失 → 名片残缺，宁可失败也不能让实体拿到半张名片
    EXPECT_EQ(1u, RunDpuQueueNamecardRestore(&param));

    param.cqDesc = mem.CqDesc();
    param.cqDesc.elemSize = 0; // 格宽必须是 CQE 的 64B，0 显然不合法
    EXPECT_EQ(1u, RunDpuQueueNamecardRestore(&param));
}

TEST_F(TestCpuTaskPipe, Ut_QueueNamecardRestore_When_ThreadHandleNull_Expect_Fail)
{
    FakeDputag mem;
    DpuQueueNamecardParam param{};
    param.sqDesc = mem.SqDesc();
    param.cqDesc = mem.CqDesc();

    // 名片没有落点：CQ 守护侧再也拿不到这张 CQ，直接判失败
    EXPECT_EQ(1u, RunDpuQueueNamecardRestore(&param));
}

// ==================== device 侧 CQ 轮询 ====================

TEST_F(TestCpuTaskPipe, Ut_DpuCqPoll_When_DepthZero_Expect_NoOp)
{
    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "dpuCqPollDepthZero";
    QueueDescriptor cqDesc{};
    cqDesc.base = 0x1000ULL; // depth == 0 的名片不可用，轮询直接返回
    EXPECT_EQ(HCCL_SUCCESS, hcomm::DpuTaskExceptionLite::HandleDpuCq(&aicpuComm, cqDesc));
    EXPECT_FALSE(aicpuComm.IsErrorReported());
}

TEST_F(TestCpuTaskPipe, Ut_DpuCqPoll_When_CqeError_Expect_ConsumeCiAndReport)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "dpuCqPollCqe";
    QueueDescriptor cqDesc = mem.CqDesc();

    DpuCqe cqe{};
    cqe.taskId = 21u;
    cqe.streamId = 3u;
    cqe.hcclErrCode = static_cast<int32_t>(HCCL_E_INTERNAL);
    mem.PokeCqElem(0, cqe);
    mem.PokeCq(1u, 0u); // pi = 1, ci = 0：一格待消费

    Hccl::DfxDfxOpInfo fakeOpInfo{};
    fakeOpInfo.cpuWaitAicpuNotifyId = 7u;
    MOCKER_CPP(&hccl::HcclCommDfxLite::GetLatestDfxOpInfo)
        .stubs()
        .will(returnValue(static_cast<const void*>(&fakeOpInfo)));
    MOCKER_CPP(&hcomm::HcclCommTaskExceptionLite::SendTaskExceptionByMBox).stubs().will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(HCCL_SUCCESS, hcomm::DpuTaskExceptionLite::HandleDpuCq(&aicpuComm, cqDesc));
    EXPECT_EQ(1u, mem.pipeCqMeta()->ci); // 消费完推进 ci，即使命中了错误
    EXPECT_TRUE(aicpuComm.IsErrorReported());
}

TEST_F(TestCpuTaskPipe, Ut_DpuCqPoll_When_NoOpInfo_Expect_KeepCqeWithoutReport)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "dpuCqPollNoOp"; // dfx_ 未初始化 → GetLatestDfxOpInfo() 为 nullptr
    QueueDescriptor cqDesc = mem.CqDesc();

    DpuCqe cqe{};
    cqe.hcclErrCode = static_cast<int32_t>(HCCL_E_INTERNAL);
    mem.PokeCqElem(0, cqe);
    mem.PokeCq(1u, 0u);

    EXPECT_EQ(HCCL_SUCCESS, hcomm::DpuTaskExceptionLite::HandleDpuCq(&aicpuComm, cqDesc));
    // 还没有算子就没有可用的上报通道：先上报后推进，上报不成则 ci 不动，回执留着下次重报
    EXPECT_EQ(0u, mem.pipeCqMeta()->ci);
    EXPECT_FALSE(aicpuComm.IsErrorReported());
}

// 失败注入的公共现场：一格错误回执待消费（pi = 1, ci = 0），
// stub 按调用序把第 g_failAtCall 次 memcpy_s 报失败。游标读写走原子操作，不可注入，
// 拷贝 CQE 内容是本函数唯一的可失败点
TEST_F(TestCpuTaskPipe, Ut_DpuCqPoll_When_CqeReadFail_Expect_MemoryError)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "dpuCqPollCqeFail";
    QueueDescriptor cqDesc = mem.CqDesc();

    DpuCqe cqe{};
    cqe.hcclErrCode = static_cast<int32_t>(HCCL_E_INTERNAL);
    mem.PokeCqElem(0, cqe);
    mem.PokeCq(1u, 0u);

    g_memcpyCallCount = 0;
    g_failAtCall = 1; // 读 cqe（pi/ci 已读出，pi != ci 进入消费）
    MOCKER(memcpy_s).stubs().will(invoke(StubMemcpyFailAt));

    EXPECT_EQ(HCCL_E_MEMORY, hcomm::DpuTaskExceptionLite::HandleDpuCq(&aicpuComm, cqDesc));
    EXPECT_EQ(0u, mem.pipeCqMeta()->ci); // 回执没读出来不算消费，ci 不动
    EXPECT_FALSE(aicpuComm.IsErrorReported());
}

TEST_F(TestCpuTaskPipe, Ut_DpuCqPoll_When_AlreadyReported_Expect_EarlyReturn)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "dpuCqPollDup";
    aicpuComm.SetErrorReported(true); // 与 task exception 上报共用同一份「已上报」状态
    QueueDescriptor cqDesc = mem.CqDesc();

    DpuCqe cqe{};
    cqe.hcclErrCode = static_cast<int32_t>(HCCL_E_INTERNAL);
    mem.PokeCqElem(0, cqe);
    mem.PokeCq(1u, 0u);

    EXPECT_EQ(HCCL_SUCCESS, hcomm::DpuTaskExceptionLite::HandleDpuCq(&aicpuComm, cqDesc));
    // 防重早退发生在读游标之前：重复轮询不二次消费、也不二次上报
    EXPECT_EQ(0u, mem.pipeCqMeta()->ci);
    EXPECT_TRUE(aicpuComm.IsErrorReported());
}

TEST_F(TestCpuTaskPipe, Ut_DpuCqPoll_When_QueueEmpty_Expect_EarlyReturn)
{
    FakeDputag mem;
    CpuTaskPipe pipe;
    ASSERT_EQ(HCCL_SUCCESS, pipe.Init(mem.Layout()));

    CollCommAicpu aicpuComm;
    aicpuComm.identifier_ = "dpuCqPollEmpty";
    QueueDescriptor cqDesc = mem.CqDesc();

    mem.PokeCq(0u, 0u); // pi == ci：无回执

    EXPECT_EQ(HCCL_SUCCESS, hcomm::DpuTaskExceptionLite::HandleDpuCq(&aicpuComm, cqDesc));
    EXPECT_EQ(0u, mem.pipeCqMeta()->ci);
    EXPECT_FALSE(aicpuComm.IsErrorReported());
}

// ==================== 队列模式与旧方案的互斥 ====================

TEST_F(TestCpuTaskPipe, Ut_QueueMode_When_Normal_Expect_LaunchedWithoutTaskService)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    std::thread sim;
    EXPECT_EQ(HCCL_SUCCESS, InitQueueMode(thread));
    StartKernelSim(thread, sim);
    EXPECT_TRUE(thread.IsKernelLaunched());
    // 新方案不构造 TaskService：这是它与旧方案在同一条 thread 上共存的前提
    EXPECT_EQ(nullptr, thread.GetLegacy()->GetTaskService());
    EXPECT_NE(nullptr, thread.GetTaskPipe());
    EXPECT_TRUE(thread.GetTaskPipe()->IsInited());

    // 二次进入幂等（判据是 dpuTaskPipe_ 本身，不是 launchMode_ 之类的状态位）
    EXPECT_EQ(HCCL_SUCCESS, InitQueueMode(thread));
    EXPECT_EQ(HCCL_SUCCESS, thread.DeInit());
    sim.join();
}

TEST_F(TestCpuTaskPipe, Ut_QueueMode_When_SetXpuDeviceFailed_Expect_NoPipeLeftBehind)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_RUNTIME));
    MOCKER(HrtResetXpuDevice).expects(never());

    CpuThread thread;
    Hccl::TaskServiceArgs args{"testCommId", 0};
    EXPECT_EQ(HCCL_E_INTERNAL, thread.InitQueueMode(args));
    // 半途失败必须把 dpuTaskPipe_ 收回：留着它会让重试的幂等判据假成功（pipe 在、kernel 没起）
    EXPECT_EQ(nullptr, thread.GetTaskPipe());
    EXPECT_FALSE(thread.IsKernelLaunched());
    EXPECT_EQ(HCCL_SUCCESS, thread.DeInit()); // 未拉起则早退，不 reset
}

TEST_F(TestCpuTaskPipe, Ut_QueueMode_When_LegacyOccupied_Expect_NotSupport)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    // 先按旧方案的入口拉起：1 thread 1 kernel，新方案必须拒绝而不是再下一个 kernel
    Hccl::TaskServiceArgs args{"testCommId", 0};
    ASSERT_EQ(HCCL_SUCCESS, thread.InitLegacy(args));
    *static_cast<uint8_t*>(thread.GetLegacy()->GetDputagAddr()) = EXIT_SIGNAL;
    EXPECT_NE(nullptr, thread.GetLegacy()->GetTaskService());

    EXPECT_EQ(HCCL_E_NOT_SUPPORT, thread.InitQueueMode(args));
    EXPECT_EQ(nullptr, thread.GetTaskPipe());
    EXPECT_EQ(HCCL_SUCCESS, thread.GetLegacy()->DeInit());
}

TEST_F(TestCpuTaskPipe, Ut_LegacyInit_When_QueueModeOccupied_Expect_NotSupport)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    std::thread sim;
    ASSERT_EQ(HCCL_SUCCESS, InitQueueMode(thread));
    StartKernelSim(thread, sim);

    // 反向判据：已被新方案占用的 thread 走旧方案的入口也必须被挡住（不新增 launchMode_ 的前提）
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, thread.InitLegacy(Hccl::TaskServiceArgs{"testCommId", 0}));
    EXPECT_EQ(nullptr, thread.GetLegacy()->GetTaskService());
    EXPECT_EQ(HCCL_SUCCESS, thread.DeInit());
    sim.join();
}

TEST_F(TestCpuTaskPipe, Ut_QueueMode_When_NotInited_Expect_DescriptorRefused)
{
    CpuThread thread;
    DpuSyncLayout layout{};
    QueueDescriptor sqDesc{};
    QueueDescriptor cqDesc{};
    uint64_t notifyHostAddr = 0;
    uint64_t notifyDevAddr = 0;
    // 未拉起队列时取不到 device 视角描述，投递方必须据此放弃
    EXPECT_EQ(HCCL_E_INTERNAL, thread.GetDevSyncLayout(layout, sqDesc, cqDesc, notifyHostAddr, notifyDevAddr));
}

TEST_F(TestCpuTaskPipe, Ut_QueueMode_When_Launched_Expect_DevSyncLayoutConsistent)
{
    MockInitDeps(DevType::DEV_TYPE_950);
    MOCKER(HrtSetXpuDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    MOCKER(HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    CpuThread thread;
    std::thread sim;
    ASSERT_EQ(HCCL_SUCCESS, InitQueueMode(thread));
    StartKernelSim(thread, sim);
    ThreadJoiner simJoiner(sim, [&thread]() {
        thread.GetTaskPipe()->RequestDestroy();
    });

    DpuSyncLayout layout{};
    QueueDescriptor sqDesc{};
    QueueDescriptor cqDesc{};
    uint64_t notifyHostAddr = 0;
    uint64_t notifyDevAddr = 0;
    ASSERT_EQ(HCCL_SUCCESS, thread.GetDevSyncLayout(layout, sqDesc, cqDesc, notifyHostAddr, notifyDevAddr));
    const DpuSyncLayout& hostLayout = thread.GetQueueLayout();
    EXPECT_EQ(hostLayout.sqDepth, layout.sqDepth);
    EXPECT_EQ(hostLayout.cqDepth, layout.cqDepth);
    EXPECT_EQ(hostLayout.elemSize, layout.elemSize);
    // device 地址 = device 基址 + (host 地址 - host 基址)。本用例的 HostRegister 桩让两个基址同值，
    // 于是名片上每个地址都应恰好等于 host 视角的对应地址（相对偏移一致这条契约的运行期形式）
    EXPECT_EQ(hostLayout.sqMeta, layout.sqMeta);
    EXPECT_EQ(hostLayout.sqElem, layout.sqElem);
    EXPECT_EQ(hostLayout.cqMeta, layout.cqMeta);
    EXPECT_EQ(hostLayout.cqElem, layout.cqElem);

    // SQ 只承载任务：格宽取 SQE 语义
    EXPECT_EQ(layout.sqElem, sqDesc.base);
    EXPECT_EQ(layout.sqDepth, sqDesc.depth);
    EXPECT_EQ(DPU_SQE_SIZE, sqDesc.elemSize);
    EXPECT_EQ(layout.sqMeta + offsetof(DpuSqMeta, pi), sqDesc.piAddr);
    EXPECT_EQ(layout.sqMeta + offsetof(DpuSqMeta, ci), sqDesc.ciAddr);

    // CQ 侧只承载回执，两个描述必须落在同一张名片的不同半区
    EXPECT_EQ(layout.cqElem, cqDesc.base);
    EXPECT_EQ(layout.cqDepth, cqDesc.depth);
    EXPECT_EQ(DPU_CQE_SIZE, cqDesc.elemSize);
    EXPECT_EQ(layout.cqMeta + offsetof(DpuCqMeta, pi), cqDesc.piAddr);
    EXPECT_EQ(layout.cqMeta + offsetof(DpuCqMeta, ci), cqDesc.ciAddr);
    EXPECT_NE(sqDesc.base, cqDesc.base);
    EXPECT_EQ(HCCL_SUCCESS, thread.DeInit());
    sim.join();
}

// ==================== 新方案挂 AICPU 引擎：device 侧实体与其登记表 ====================

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_When_ThreadTypeCpu_Expect_EntityWithFakeRes)
{
    bool isCpuDeviceThread = false;
    std::shared_ptr<Thread> entity;
    ASSERT_EQ(
        HCCL_SUCCESS,
        MakeCpuDeviceThread(
            THREAD_TYPE_CPU, "testCommId", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950), isCpuDeviceThread, entity));

    ASSERT_TRUE(isCpuDeviceThread);
    ASSERT_NE(nullptr, entity);
    // 复用 fakeDeviceRes_ 是本案的关键契约：L2/L3 的 DFX/Cache 注册与 GetAllThread() 消费方全按它分流
    EXPECT_TRUE(entity->IsFakeDeviceRes());
    EXPECT_EQ(nullptr, entity->GetStream());
    EXPECT_EQ(nullptr, entity->GetStreamLitePtr());
    EXPECT_EQ(0u, entity->GetNotifyNum());
    EXPECT_TRUE(entity->GetUniqueId().empty()); // 身份走参数块 out-of-band 字段，不进序列化 blob

    const auto handle = reinterpret_cast<ThreadHandle>(entity.get());
    EXPECT_EQ(entity.get(), FindCpuDeviceThread(handle).get());
    EXPECT_EQ(std::string("testCommId"), static_cast<CpuDeviceThread*>(entity.get())->GetCommId());

    EXPECT_TRUE(EraseCpuDeviceThread(handle));
    EXPECT_EQ(nullptr, FindCpuDeviceThread(handle));
}

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_When_ThreadTypeTs_Expect_NotDpuThread)
{
    bool isCpuDeviceThread = true;
    std::shared_ptr<Thread> entity;
    // 旧批次（TS/未置类型）必须原样走 AicpuTsThread：这里只断言「不是 DPU 线程」，不代造实体
    ASSERT_EQ(
        HCCL_SUCCESS,
        MakeCpuDeviceThread(
            THREAD_TYPE_TS, "testCommId", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950), isCpuDeviceThread, entity));
    EXPECT_FALSE(isCpuDeviceThread);
    EXPECT_EQ(nullptr, entity);
}

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_When_DataPlaneOps_Expect_NotSupport)
{
    CpuDeviceThread entity("testCommId", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950));
    EXPECT_TRUE(entity.IsFakeDeviceRes());
    EXPECT_EQ(HCCL_SUCCESS, entity.Init());
    EXPECT_EQ(HCCL_SUCCESS, entity.DeInit());
    EXPECT_EQ(HCCL_SUCCESS, entity.SupplementNotify(1)); // 无 notify 可补，按空操作处理而非报错
    EXPECT_EQ(nullptr, entity.GetNotify(0));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, entity.LocalNotifyRecord(0u));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, entity.LocalNotifyWait(0u));
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, entity.LocalReduce(nullptr, nullptr, 0, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
}

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_When_IsDeviceA5_Expect_TrueFor950)
{
    EXPECT_TRUE(CpuDeviceThread("c", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950)).IsDeviceA5());
    EXPECT_FALSE(CpuDeviceThread("c", 0, static_cast<uint32_t>(DevType::DEV_TYPE_910B)).IsDeviceA5());
}

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_LocalCopy_When_NamecardGiven_Expect_SqeSubmitted)
{
    FakeDputag mem;
    mem.PokeSqPi(0);
    mem.PokeSqCi(0);
    bool isCpuDeviceThread = false;
    std::shared_ptr<Thread> entity;
    ASSERT_EQ(
        HCCL_SUCCESS,
        MakeCpuDeviceThread(
            THREAD_TYPE_CPU, "comm_lc", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950), isCpuDeviceThread, entity));
    ASSERT_TRUE(isCpuDeviceThread);

    EXPECT_EQ(HCCL_E_INTERNAL, entity->LocalCopy(nullptr, nullptr, 0));

    DpuQueueNamecardParam param{};
    param.cqDesc = mem.CqDesc();
    param.sqDesc = mem.SqDesc();
    param.layout = mem.Layout();
    param.threadHandle = reinterpret_cast<ThreadHandle>(entity.get());
    ASSERT_EQ(0u, RunDpuQueueNamecardRestore(&param));

    void* dst = reinterpret_cast<void*>(0x1000ULL);
    const void* src = reinterpret_cast<const void*>(0x2000ULL);
    EXPECT_EQ(HCCL_SUCCESS, entity->LocalCopy(dst, src, 64));

    DpuSqe got = mem.PeekSqElem(0);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_LOCAL_COPY), got.opKind);
    EXPECT_EQ(0x1000ULL, got.u.localCopy.dst);
    EXPECT_EQ(0x2000ULL, got.u.localCopy.src);
    EXPECT_EQ(64U, got.u.localCopy.len);
    EXPECT_EQ(0U, got.taskId);

    EXPECT_TRUE(EraseCpuDeviceThread(param.threadHandle));
}

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_NotifyWait_When_NamecardGiven_Expect_SqeSubmitted)
{
    FakeDputag mem;
    mem.PokeSqPi(0);
    mem.PokeSqCi(0);
    bool isCpuDeviceThread = false;
    std::shared_ptr<Thread> entity;
    ASSERT_EQ(
        HCCL_SUCCESS,
        MakeCpuDeviceThread(
            THREAD_TYPE_CPU, "comm_nw", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950), isCpuDeviceThread, entity));
    ASSERT_TRUE(isCpuDeviceThread);

    // 越界下标在构造 SQE 之前就被挡住：不占 taskId、不推进 pi
    EXPECT_EQ(HCCL_E_PARA, entity->LocalNotifyWait(NOTIFY_TEST_NUM, 2));
    EXPECT_EQ(HCCL_E_PARA, entity->LocalNotifyWait(3, 7));

    DpuQueueNamecardParam param{};
    param.cqDesc = mem.CqDesc();
    param.sqDesc = mem.SqDesc();
    param.layout = mem.Layout();
    param.threadHandle = reinterpret_cast<ThreadHandle>(entity.get());
    param.notifyNum = NOTIFY_TEST_NUM;
    ASSERT_EQ(0u, RunDpuQueueNamecardRestore(&param));

    EXPECT_EQ(HCCL_SUCCESS, entity->LocalNotifyWait(3, 7));

    DpuSqe got = mem.PeekSqElem(0);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_THREAD_NOTIFY_WAIT), got.opKind);
    EXPECT_EQ(3u, got.u.threadNotifyWait.notifyIdx);
    EXPECT_EQ(7u, got.u.threadNotifyWait.timeOut);
    EXPECT_EQ(0u, got.taskId);
    EXPECT_EQ(1u, mem.pipeSqMeta()->pi);

    EXPECT_TRUE(EraseCpuDeviceThread(param.threadHandle));
}

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_NotifyRecord_When_DstVariants_Expect_MatchedSqeOrError)
{
    FakeDputag mem;
    mem.PokeSqPi(0);
    mem.PokeSqCi(0);
    bool isCpuDeviceThread = false;
    std::shared_ptr<Thread> entity;
    ASSERT_EQ(
        HCCL_SUCCESS,
        MakeCpuDeviceThread(
            THREAD_TYPE_CPU, "comm_nr", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950), isCpuDeviceThread, entity));
    ASSERT_TRUE(isCpuDeviceThread);

    DpuQueueNamecardParam param{};
    param.cqDesc = mem.CqDesc();
    param.sqDesc = mem.SqDesc();
    param.layout = mem.Layout();
    param.threadHandle = reinterpret_cast<ThreadHandle>(entity.get());
    ASSERT_EQ(0u, RunDpuQueueNamecardRestore(&param));

    // 空句柄：参数错误直接失败，不占 taskId
    EXPECT_EQ(HCCL_E_PTR, entity->LocalNotifyRecord(0, 5));

    CpuDeviceThread bareDst("bareDst", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950));
    EXPECT_EQ(HCCL_E_PARA, entity->LocalNotifyRecord(reinterpret_cast<ThreadHandle>(&bareDst), 5));

    // 对端是 DPU 线程：烧对端的 host 视角绝对地址（notifyHostAddr + dstNotifyIdx）
    uint8_t devShmem[NOTIFY_TEST_NUM] = {0};
    uint8_t hostShmem[NOTIFY_TEST_NUM] = {0};
    CpuDeviceThread dpuDst("dpuDst", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950));
    dpuDst.SetQueueNamecard(
        mem.SqDesc(), mem.CqDesc(), NOTIFY_TEST_NUM, reinterpret_cast<uint64_t>(hostShmem),
        reinterpret_cast<uint64_t>(devShmem));
    EXPECT_EQ(HCCL_SUCCESS, entity->LocalNotifyRecord(reinterpret_cast<ThreadHandle>(&dpuDst), 5));

    DpuSqe got = mem.PeekSqElem(0);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_THREAD_NOTIFY_RECORD_TO_DPU), got.opKind);
    EXPECT_EQ(reinterpret_cast<uint64_t>(hostShmem + 5), got.u.threadNotifyRecordToDpu.addr);
    EXPECT_EQ(0u, got.taskId);

    // 对端是 NPU 线程：地址口径不适用，退化为 notifyId
    NpuPeerThread npuDst;
    EXPECT_EQ(HCCL_SUCCESS, entity->LocalNotifyRecord(reinterpret_cast<ThreadHandle>(&npuDst), 3));

    got = mem.PeekSqElem(1);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_THREAD_NOTIFY_RECORD_TO_NPU), got.opKind);
    EXPECT_EQ(PEER_NOTIFY_ID, got.u.threadNotifyRecordToNpu.notifyId);
    EXPECT_EQ(1u, got.taskId); // taskId 跨两次提交连续分配
    EXPECT_EQ(2u, mem.pipeSqMeta()->pi);

    EXPECT_TRUE(EraseCpuDeviceThread(param.threadHandle));
}

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_ChannelOps_When_NamecardGiven_Expect_SqeFields)
{
    FakeDputag mem;
    mem.PokeSqPi(0);
    mem.PokeSqCi(0);
    bool isCpuDeviceThread = false;
    std::shared_ptr<Thread> entity;
    ASSERT_EQ(
        HCCL_SUCCESS,
        MakeCpuDeviceThread(
            THREAD_TYPE_CPU, "comm_ch", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950), isCpuDeviceThread, entity));
    ASSERT_TRUE(isCpuDeviceThread);

    DpuQueueNamecardParam param{};
    param.cqDesc = mem.CqDesc();
    param.sqDesc = mem.SqDesc();
    param.layout = mem.Layout();
    param.threadHandle = reinterpret_cast<ThreadHandle>(entity.get());
    ASSERT_EQ(0u, RunDpuQueueNamecardRestore(&param));

    const ChannelHandle channel = 0x3000ULL;
    void* dst = reinterpret_cast<void*>(0x1000ULL);
    const void* src = reinterpret_cast<const void*>(0x2000ULL);

    EXPECT_EQ(HCCL_SUCCESS, entity->LocalChannelNotifyWait(channel, 5, 9));
    DpuSqe got = mem.PeekSqElem(0);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_CHANNEL_NOTIFY_WAIT), got.opKind);
    EXPECT_EQ(channel, got.u.channelNotifyWait.channelHandle);
    EXPECT_EQ(5u, got.u.channelNotifyWait.notifyIdx);
    EXPECT_EQ(9u, got.u.channelNotifyWait.timeOut);
    EXPECT_EQ(0u, got.taskId);

    EXPECT_EQ(HCCL_SUCCESS, entity->LocalChannelNotifyRecord(channel, 5));
    got = mem.PeekSqElem(1);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_CHANNEL_NOTIFY_RECORD), got.opKind);
    EXPECT_EQ(channel, got.u.channelNotifyRecord.channelHandle);
    EXPECT_EQ(5u, got.u.channelNotifyRecord.notifyIdx);
    EXPECT_EQ(1u, got.taskId);

    EXPECT_EQ(HCCL_SUCCESS, entity->LocalChannelWriteWithNotifyNbi(channel, dst, src, 64, 5));
    got = mem.PeekSqElem(2);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_CHANNEL_WRITE_WITH_NOTIFY_NBI), got.opKind);
    EXPECT_EQ(channel, got.u.channelWriteWithNotify.channelHandle);
    EXPECT_EQ(0x1000ULL, got.u.channelWriteWithNotify.dstAddr);
    EXPECT_EQ(0x2000ULL, got.u.channelWriteWithNotify.srcAddr);
    EXPECT_EQ(64U, got.u.channelWriteWithNotify.len);
    EXPECT_EQ(5u, got.u.channelWriteWithNotify.notifyIdx);
    EXPECT_EQ(2u, got.taskId);

    EXPECT_EQ(HCCL_SUCCESS, entity->LocalChannelDrain(channel));
    got = mem.PeekSqElem(3);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_CHANNEL_DRAIN), got.opKind);
    EXPECT_EQ(channel, got.u.channelDrainOrFence.channelHandle);
    EXPECT_EQ(3u, got.taskId);

    EXPECT_EQ(HCCL_SUCCESS, entity->LocalChannelFence(channel));
    got = mem.PeekSqElem(4);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_CHANNEL_FENCE), got.opKind);
    EXPECT_EQ(channel, got.u.channelDrainOrFence.channelHandle);
    EXPECT_EQ(4u, got.taskId);

    EXPECT_EQ(HCCL_SUCCESS, entity->LocalCommFence());
    got = mem.PeekSqElem(5);
    EXPECT_EQ(static_cast<uint16_t>(DPU_OP_COMM_FENCE), got.opKind);
    EXPECT_EQ(5u, got.taskId); // taskId 从 0 起连续分配

    EXPECT_EQ(6u, mem.pipeSqMeta()->pi);

    EXPECT_TRUE(EraseCpuDeviceThread(param.threadHandle));
}

TEST_F(TestCpuTaskPipe, Ut_DpuEntity_NotifyShmAddr_When_IdxAndNamecard_Expect_AddrOrError)
{
    FakeDputag mem;
    bool isCpuDeviceThread = false;
    std::shared_ptr<Thread> entity;
    ASSERT_EQ(
        HCCL_SUCCESS,
        MakeCpuDeviceThread(
            THREAD_TYPE_CPU, "comm_na", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950), isCpuDeviceThread, entity));
    ASSERT_TRUE(isCpuDeviceThread);

    uint64_t shmAddr = 0;
    EXPECT_EQ(HCCL_E_PARA, entity->GetNotifyDevShmAddr(0, shmAddr));
    EXPECT_EQ(HCCL_E_PARA, entity->GetNotifyHostShmAddr(0, shmAddr));

    uint8_t devShmem[NOTIFY_TEST_NUM] = {0};
    uint8_t hostShmem[NOTIFY_TEST_NUM] = {0};
    DpuQueueNamecardParam param{};
    param.cqDesc = mem.CqDesc();
    param.sqDesc = mem.SqDesc();
    param.layout = mem.Layout();
    param.threadHandle = reinterpret_cast<ThreadHandle>(entity.get());
    param.notifyDevAddr = reinterpret_cast<uint64_t>(devShmem);
    param.notifyHostAddr = reinterpret_cast<uint64_t>(hostShmem);
    param.notifyNum = NOTIFY_TEST_NUM;
    ASSERT_EQ(0u, RunDpuQueueNamecardRestore(&param));

    // 名片的两个 notify 地址各自随下标偏移一字节
    EXPECT_EQ(HCCL_SUCCESS, entity->GetNotifyDevShmAddr(7, shmAddr));
    EXPECT_EQ(reinterpret_cast<uint64_t>(devShmem + 7), shmAddr);
    EXPECT_EQ(HCCL_SUCCESS, entity->GetNotifyHostShmAddr(9, shmAddr));
    EXPECT_EQ(reinterpret_cast<uint64_t>(hostShmem + 9), shmAddr);

    // 越界下标两侧同判参数错误
    EXPECT_EQ(HCCL_E_PARA, entity->GetNotifyDevShmAddr(NOTIFY_TEST_NUM, shmAddr));
    EXPECT_EQ(HCCL_E_PARA, entity->GetNotifyHostShmAddr(NOTIFY_TEST_NUM, shmAddr));

    EXPECT_TRUE(EraseCpuDeviceThread(param.threadHandle));
}

TEST_F(TestCpuTaskPipe, Ut_QueueNamecardRestore_When_ThreadHandleGiven_Expect_NamecardFilled)
{
    FakeDputag mem;
    bool isCpuDeviceThread = false;
    std::shared_ptr<Thread> entity;
    ASSERT_EQ(
        HCCL_SUCCESS,
        MakeCpuDeviceThread(
            THREAD_TYPE_CPU, "comm_k", 0, static_cast<uint32_t>(DevType::DEV_TYPE_950), isCpuDeviceThread, entity));
    ASSERT_TRUE(isCpuDeviceThread);

    DpuQueueNamecardParam param{};
    param.cqDesc = mem.CqDesc();
    param.sqDesc = mem.SqDesc();
    param.layout = mem.Layout();
    param.threadHandle = reinterpret_cast<ThreadHandle>(entity.get());

    auto* dpuEntity = static_cast<CpuDeviceThread*>(entity.get());
    EXPECT_FALSE(dpuEntity->IsNamecardReady());
    EXPECT_FALSE(dpuEntity->GetSqSubmitter().IsInited());
    EXPECT_EQ(0u, RunDpuQueueNamecardRestore(&param));
    EXPECT_TRUE(dpuEntity->IsNamecardReady());
    EXPECT_EQ(param.cqDesc.base, dpuEntity->GetCqDesc().base);
    EXPECT_EQ(param.cqDesc.depth, dpuEntity->GetCqDesc().depth);
    // 名片一到手，SQ 生产侧即可用：AICPU 编排此后按 handle → 实体 → submitter 下发任务
    EXPECT_TRUE(dpuEntity->GetSqSubmitter().IsInited());
    EXPECT_EQ(param.sqDesc.base, dpuEntity->GetSqSubmitter().GetSqDesc().base);

    EXPECT_TRUE(EraseCpuDeviceThread(param.threadHandle));
}

TEST_F(TestCpuTaskPipe, Ut_QueueNamecardRestore_When_ThreadHandleUnknown_Expect_OkWithoutNamecard)
{
    FakeDputag mem;
    // device 侧会先解引用句柄取 ABI 头，句柄必须是有效实体地址；这里只让 compatThreadHandle
    // 不在登记表里，模拟销毁与投递竞争时名片无处可落
    HcommThreadEntity entity{};
    entity.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(entity), 0};
    entity.compatThreadHandle = 0x1234ULL;

    DpuQueueNamecardParam param{};
    param.sqDesc = mem.SqDesc();
    param.cqDesc = mem.CqDesc();
    param.threadHandle = reinterpret_cast<ThreadHandle>(&entity);

    EXPECT_EQ(0u, RunDpuQueueNamecardRestore(&param)); // 名片落空，但不判投递失败
}

// ==================== 新方案的 host→device 类型判据与线程分派 ====================

TEST_F(TestCpuTaskPipe, Ut_ConvertTable_When_AicpuCpuPair_Expect_HostNotifyOnline)
{
    NotifyLoadType loadType = NotifyLoadType::DEVICE_NOTIFY;
    StreamType streamType = StreamType::STREAM_TYPE_DEVICE;
    // 无 TS 真流：取 DEVICE 会被读方误判为「有 device 流」，故与 (CPU,CPU) 同口径
    ASSERT_EQ(HCCL_SUCCESS, GetNotifyLoadType(COMM_ENGINE_AICPU, THREAD_TYPE_CPU, loadType));
    EXPECT_EQ(NotifyLoadType::HOST_NOTIFY, loadType);
    ASSERT_EQ(HCCL_SUCCESS, GetStreamType(COMM_ENGINE_AICPU, THREAD_TYPE_CPU, streamType));
    EXPECT_EQ(StreamType::STREAM_TYPE_ONLINE, streamType);

    // 对照：AICPU 的 TS 线程仍取 DEVICE，两条口径不得互相污染
    ASSERT_EQ(HCCL_SUCCESS, GetStreamType(COMM_ENGINE_AICPU, THREAD_TYPE_TS, streamType));
    EXPECT_EQ(StreamType::STREAM_TYPE_DEVICE, streamType);
}

TEST_F(TestCpuTaskPipe, Ut_CreateThread_When_ThreadTypeCpu_Expect_CpuThreadForBothEngines)
{
    for (CommEngine engine : {COMM_ENGINE_CPU, COMM_ENGINE_AICPU}) {
        std::shared_ptr<Thread> thread;
        ASSERT_EQ(
            HCCL_SUCCESS,
            CreateThread(
                engine, THREAD_TYPE_CPU, StreamType::STREAM_TYPE_ONLINE, 0, NotifyLoadType::HOST_NOTIFY, thread));
        ASSERT_NE(nullptr, thread);
        EXPECT_NE(nullptr, std::dynamic_pointer_cast<CpuThread>(thread));
        // 两条路的host壳同型，device 侧靠这一字段决定建哪类实体
        EXPECT_EQ(engine, thread->GetCommEngine());
        EXPECT_EQ(THREAD_TYPE_CPU, thread->GetThreadType());
    }
}
