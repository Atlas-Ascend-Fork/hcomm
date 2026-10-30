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
#include <mockcpp/mockcpp.hpp>
#include "adapter_rts_common.h"
#include "new/hccl_primitive_local.h"
#define private public
#include "cpu_ts_thread.h"
#include "aicpu_ts_thread.h"
#undef private

using namespace hccl;

static Stream* expectedStream = nullptr;
static LocalNotify* expectedNotify = nullptr;
static HcclResult recordResult = HCCL_SUCCESS;

static HcclResult RecordNotify(StreamHandle stream, aclrtNotify notify)
{
    EXPECT_EQ(stream, expectedStream);
    EXPECT_EQ(notify, expectedNotify);
    return recordResult;
}

static HcclResult PostNotify(LocalNotify* notify, Stream& stream)
{
    EXPECT_EQ(&stream, expectedStream);
    EXPECT_EQ(notify, expectedNotify);
    return recordResult;
}

class UtCpuHcommThreadNotifyRecordOnThread : public testing::Test {
protected:
    void SetUp() override
    {
        source.stream_.reset(new Stream());
        destination = std::make_shared<AicpuTsThread>(StreamType::STREAM_TYPE_DEVICE, 1, NotifyLoadType::DEVICE_NOTIFY);
        destination->notifys_.emplace_back(new LocalNotify());
        expectedStream = source.stream_.get();
        expectedNotify = destination->notifys_[0].get();
        recordResult = HCCL_SUCCESS;
    }

    void TearDown() override
    {
        destination.reset();
        GlobalMockObject::verify();
    }

    void MockRecord(DevType type)
    {
        source.devType_ = type;
        // A5/A3 分支由全局芯片类型（hrtGetDeviceType）决定，与 thread 自身 devType_ 无关。
        MOCKER(hrtGetDeviceType).stubs().with(outBound(type)).will(returnValue(HCCL_SUCCESS));
        if (type == DevType::DEV_TYPE_950) {
            MOCKER_CPP(static_cast<HcclResult (LocalNotify::*)(Stream&)>(&LocalNotify::Post))
                .expects(once())
                .will(invoke(PostNotify));
            MOCKER_CPP(&LocalNotify::GetNotifyData)
                .stubs()
                .with(outBound(HcclSignalInfo{}))
                .will(returnValue(HCCL_SUCCESS));
            MOCKER_CPP(&Thread::ReportHostNotifyRecordTask).stubs().will(returnValue(HCCL_SUCCESS));
        } else {
            MOCKER(HcclLocalNotifyRecord).expects(once()).will(invoke(RecordNotify));
        }
    }

    void CheckHostHandle(DevType type)
    {
        MockRecord(type);
        // Host 原语使用调用方已转换的 Host 句柄，不依赖资源管理表。
        ThreadHandle sourceHandle = reinterpret_cast<ThreadHandle>(&source);
        EXPECT_EQ(
            HcommThreadNotifyRecordOnThread(sourceHandle, reinterpret_cast<ThreadHandle>(destination.get()), 0),
            HCCL_SUCCESS);
    }

    CpuTsThread source{StreamType::STREAM_TYPE_ONLINE, 0, NotifyLoadType::DEVICE_NOTIFY};
    std::shared_ptr<AicpuTsThread> destination;
};

TEST_F(UtCpuHcommThreadNotifyRecordOnThread, A3RecordsWithHostHandleWithoutResourceLookup)
{
    CheckHostHandle(DevType::DEV_TYPE_910_93);
}

TEST_F(UtCpuHcommThreadNotifyRecordOnThread, A5RecordsWithHostHandleWithoutResourceLookup)
{
    CheckHostHandle(DevType::DEV_TYPE_950);
}
