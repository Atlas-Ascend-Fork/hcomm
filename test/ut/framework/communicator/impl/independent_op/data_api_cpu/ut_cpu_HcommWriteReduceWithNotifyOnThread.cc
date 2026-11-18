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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_primitives.h"

#include "host_cpu_roce_channel.h"

using namespace hccl;

class UtCpuHcommWriteReduceWithNotifyOnThread : public testing::Test {
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override { GlobalMockObject::verify(); }

    // thread is unused by the builtin write-reduce-with-notify path for non-null threads on 950.
    ThreadHandle thread = static_cast<ThreadHandle>(0x01);
    EndpointHandle epHandle = reinterpret_cast<void*>(0x01);
    HcommChannelDesc channelDesc{};
    hcomm::HostCpuRoceChannel channelOnHost{epHandle, channelDesc};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&channelOnHost);
    char srcBuf[32]{};
    char dstBuf[32]{};
    void* src = srcBuf;
    void* dst = dstBuf;
    uint64_t count = 8; // FP32 => len = 32B
    HcommDataType dataType = HCOMM_DATA_TYPE_FP32;
    HcommReduceOp reduceOp = HCOMM_REDUCE_SUM;
    uint32_t remoteNotifyIdx = 0;
    int32_t res{HCCL_E_RESERVED};
    DevType t950 = DevType::DEV_TYPE_950;
    DevType t910C = DevType::DEV_TYPE_910_93;
};

// 950

TEST_F(
    UtCpuHcommWriteReduceWithNotifyOnThread,
    Ut_HcommWriteReduceWithNotifyOnThread_When_950_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::WriteReduceWithNotify)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    res = HcommWriteReduceWithNotifyOnThread(thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtCpuHcommWriteReduceWithNotifyOnThread,
    Ut_HcommWriteReduceWithNotifyOnThread_When_950_Thread_IsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::WriteReduceWithNotify)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    res = HcommWriteReduceWithNotifyOnThread(0, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtCpuHcommWriteReduceWithNotifyOnThread,
    Ut_HcommWriteReduceWithNotifyOnThread_When_950_Channel_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    res = HcommWriteReduceWithNotifyOnThread(thread, 0, dst, src, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtCpuHcommWriteReduceWithNotifyOnThread,
    Ut_HcommWriteReduceWithNotifyOnThread_When_Src_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommWriteReduceWithNotifyOnThread(thread, channel, dst, nullptr, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtCpuHcommWriteReduceWithNotifyOnThread,
    Ut_HcommWriteReduceWithNotifyOnThread_When_Dst_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommWriteReduceWithNotifyOnThread(thread, channel, nullptr, src, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtCpuHcommWriteReduceWithNotifyOnThread,
    Ut_HcommWriteReduceWithNotifyOnThread_When_950_WriteReduceWithNotify_Fails_Expect_ErrorCodePropagated)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::WriteReduceWithNotify)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    res = HcommWriteReduceWithNotifyOnThread(thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}

TEST_F(
    UtCpuHcommWriteReduceWithNotifyOnThread,
    Ut_HcommWriteReduceWithNotifyOnThread_When_950_WriteReduceWithNotify_Again_Expect_AgainPropagated)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::WriteReduceWithNotify)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_AGAIN));
    res = HcommWriteReduceWithNotifyOnThread(thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_AGAIN);
}

// 910C

TEST_F(
    UtCpuHcommWriteReduceWithNotifyOnThread,
    Ut_HcommWriteReduceWithNotifyOnThread_When_910C_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t910C)).will(returnValue(HCCL_SUCCESS));
    res = HcommWriteReduceWithNotifyOnThread(thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}
