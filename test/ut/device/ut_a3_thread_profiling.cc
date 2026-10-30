/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "aicpu_ts_thread.h"
#include "hcom_device_profiling.h"
#include "device/inc/profiling_manager_device.h"

namespace {
using namespace hccl;

class A3ProfilingThread : public AicpuTsThread {
public:
    explicit A3ProfilingThread(Stream* stream) : AicpuTsThread(std::string()), stream_(stream) {}
    Stream* GetStream() const override { return stream_; }

private:
    Stream* stream_;
};

class A3ThreadProfilingTest : public testing::Test {
protected:
    void SetUp() override
    {
        MOCKER(dfx::ProfilingManager::IsProfL0On).stubs().will(returnValue(false));
        MOCKER(dfx::ProfilingManager::IsProfL1On).stubs().will(returnValue(true));
        MOCKER(dfx::ProfilingManager::GetProfL1State).stubs().will(returnValue(true));
        sqeContext_ = std::make_unique<HcclSqeContext>();
        sqeContext_->buffer.tailSqeTaskId = 9;
        stream_.sqeContext_ = sqeContext_.get();
        thread_ = std::make_unique<A3ProfilingThread>(&stream_);
        rawHandle_ = reinterpret_cast<ThreadHandle>(thread_.get());
        entity_.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(HcommThreadEntity), 0};
        entity_.compatThreadHandle = rawHandle_;
        wrappedHandle_ = reinterpret_cast<ThreadHandle>(&entity_);
    }

    void TearDown() override { GlobalMockObject::verify(); }

    std::unique_ptr<HcclSqeContext> sqeContext_;
    Stream stream_;
    std::unique_ptr<A3ProfilingThread> thread_;
    HcommThreadEntity entity_{};
    ThreadHandle rawHandle_{};
    ThreadHandle wrappedHandle_{};
};

TEST_F(A3ThreadProfilingTest, WrappedAndRawHandlesSupportProfiling)
{
    MOCKER(dfx::ProfilingManager::ReportMainStreamTask)
        .expects(exactly(2))
        .with(any(), eq(uint16_t{9}), eq(uint16_t{0}))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(dfx::ProfilingManager::ReportMainStreamTask)
        .expects(exactly(2))
        .with(any(), eq(uint16_t{8}), eq(uint16_t{1}))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(dfx::ProfilingManager::UpdateStartReportSqeIdx).expects(exactly(4)).will(returnValue(HCCL_SUCCESS));
    MOCKER(dfx::ProfilingManager::ReportTaskInfo)
        .expects(exactly(2))
        .with(any(), eq(static_cast<void*>(sqeContext_.get())))
        .will(returnValue(HCCL_SUCCESS));

    for (ThreadHandle handle : {wrappedHandle_, rawHandle_}) {
        EXPECT_EQ(HcommProfilingReportMainStreamAndFirstTask(handle), HCCL_SUCCESS);
        EXPECT_EQ(HcommProfilingReportMainStreamAndLastTask(handle), HCCL_SUCCESS);
    }
    ThreadHandle handles[] = {wrappedHandle_, rawHandle_};
    EXPECT_EQ(HcommProfilingInit(handles, 2), HCCL_SUCCESS);
    EXPECT_EQ(handles[0], wrappedHandle_);
    EXPECT_EQ(handles[1], rawHandle_);
    EXPECT_EQ(HcommProfilingEnd(handles, 2), HCCL_SUCCESS);
    EXPECT_EQ(handles[0], wrappedHandle_);
    EXPECT_EQ(handles[1], rawHandle_);
    EXPECT_EQ(entity_.compatThreadHandle, rawHandle_);
    EXPECT_EQ(entity_.sqContextAddr, nullptr);
}

TEST_F(A3ThreadProfilingTest, RejectsUninitializedWrapperBeforeAccessingThread)
{
    entity_.compatThreadHandle = 0;
    EXPECT_EQ(HcommProfilingReportMainStreamAndFirstTask(wrappedHandle_), HCCL_E_INTERNAL);
    EXPECT_EQ(HcommProfilingReportMainStreamAndLastTask(wrappedHandle_), HCCL_E_INTERNAL);
    EXPECT_EQ(HcommProfilingEnd(&wrappedHandle_, 1), HCCL_E_INTERNAL);
    EXPECT_EQ(HcommProfilingInit(&wrappedHandle_, 1), HCCL_E_INTERNAL);
}
} // namespace
