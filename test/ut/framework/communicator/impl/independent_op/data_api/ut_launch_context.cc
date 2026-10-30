/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file launch_context.cc
 * @brief HCCL LaunchContext相关接口单元测试文件
 *
 * 本文件用于测试HCCL通信库中的LaunchContext相关接口，该接口用于启动批量/单次操作模式。
 * 批量操作模式允许将多个通信操作合并为一个批次执行，以提高通信效率。
 *
 * 测试的接口：LaunchContext相关接口
 * - 功能：启动批量操作模式，开始收集通信操作到批次中
 * - 参数：batchTag（批次标签，用于标识不同的批量操作）
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#include "launch_context.h"
#undef private
#include "adapter_rts_common.h"
#include "aicpu_ts_thread.h"

extern HcclResult CommTaskLaunch(ThreadHandle* threads, uint32_t threadNum); // host ffts+或aicpu stars使用
extern HcclResult DispatchAllStreams(const ThreadHandle* threads, uint32_t threadNum);

/**
 * @class UtAicpuTsHcommBatchModeStart
 * @brief HcommBatchModeStart接口测试类
 *
 * 该测试类用于验证HcommBatchModeStart接口的基本功能。
 * 测试场景包括正常调用启动批量操作。
 */
class UtLaunchContext : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }
};

/**
 * @test Ut_HcommBatchModeStart_When_Normal_Expect_ReturnHCCL_SUCCESS
 * @brief 测试正常情况下，启动批量操作应返回成功
 *
 * 测试场景：传入有效的批次标签，调用批量模式启动接口
 * 预期结果：返回HCCL_SUCCESS，表示批量操作成功启动
 */
TEST_F(UtLaunchContext, Ut_HandleEagerMode_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER(CommTaskLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    LaunchContext ctx;
    const char* batchTag = "test_batch";
    ctx.SetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_BATCH);

    ThreadHandle thread = 0xff;
    ctx.AddThreadWithTag(thread);
    ctx.AddThread(thread);
    EXPECT_EQ(ctx.HandleEagerMode(), HCCL_SUCCESS);
    EXPECT_EQ(ctx.HandleClear(), HCCL_SUCCESS);
}

TEST_F(UtLaunchContext, Ut_HandleDispatchAllStreams_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER(DispatchAllStreams).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    LaunchContext ctx;
    const char* batchTag = "test_batch";
    ctx.SetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_BATCH);

    ThreadHandle thread = 0xff;
    ctx.AddThreadWithTag(thread);
    ctx.AddThread(thread);
    EXPECT_EQ(ctx.HandleDispatchAllStreams(), HCCL_SUCCESS);
    EXPECT_EQ(ctx.HandleClear(), HCCL_SUCCESS);
}

class CountingCtxThread : public hccl::AicpuTsThread {
public:
    CountingCtxThread() : AicpuTsThread(hccl::StreamType::STREAM_TYPE_DEVICE, 0, hccl::NotifyLoadType::DEVICE_NOTIFY) {}
    bool IsDeviceA5() const override { return true; }
    void LaunchTask() const override { ++launchCount; }
    void TryLaunchTask() const override { ++dispatchCount; }
    mutable uint32_t launchCount{0};
    mutable uint32_t dispatchCount{0};
};

TEST_F(UtLaunchContext, BatchLaunchAndDispatchPreserveWrappedAndRawHandles)
{
    // CommTaskLaunch/DispatchAllStreams 内部按全局芯片类型走 A5(LaunchTask)/A3(stream) 分支，
    // 未 mock 时 hrtGetDeviceType 取数失败会落入 A3 分支导致用例失败。
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    CountingCtxThread first;
    CountingCtxThread second;
    HcommThreadEntity ctx{};
    ctx.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(HcommThreadEntity), 0};
    ctx.compatThreadHandle = reinterpret_cast<ThreadHandle>(&first);
    const ThreadHandle ctxHandle = reinterpret_cast<ThreadHandle>(&ctx);
    const ThreadHandle rawHandle = reinterpret_cast<ThreadHandle>(&second);
    ThreadHandle handles[]{ctxHandle, rawHandle};
    EXPECT_EQ(CommTaskLaunch(handles, 2), HCCL_SUCCESS);
    EXPECT_EQ(DispatchAllStreams(handles, 2), HCCL_SUCCESS);
    EXPECT_EQ(first.launchCount, 1);
    EXPECT_EQ(second.launchCount, 1);
    EXPECT_EQ(first.dispatchCount, 1);
    EXPECT_EQ(second.dispatchCount, 1);
    EXPECT_EQ(handles[0], ctxHandle);
    EXPECT_EQ(handles[1], rawHandle);
}
