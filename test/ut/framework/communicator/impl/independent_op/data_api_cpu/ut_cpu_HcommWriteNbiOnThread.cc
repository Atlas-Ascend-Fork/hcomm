/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class UtCpuHcommWriteNbiOnThread : public testing::Test {
protected:
    virtual void TearDown() override { GlobalMockObject::verify(); }

    // thread is unused by the builtin branch of HcommWriteNbiOnThread, any value is acceptable.
    ThreadHandle thread = static_cast<ThreadHandle>(0x01);
    EndpointHandle epHandle = reinterpret_cast<void*>(0x01);
    HcommChannelDesc channelDesc{};
    hcomm::HostCpuRoceChannel channelOnHost{epHandle, channelDesc};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&channelOnHost);
    char srcBuf[8]{};
    char dstBuf[8]{};
    void* src = srcBuf;
    void* dst = dstBuf;
    uint64_t len = sizeof(srcBuf);
    int32_t res{HCCL_E_RESERVED};
    DevType t950 = DevType::DEV_TYPE_950;
    DevType t910C = DevType::DEV_TYPE_910_93;
};

// 950：内建句柄（无 PLUGIN_HANDLE_FLAG）经 C API 内置分支分发到 Channel 虚函数

TEST_F(UtCpuHcommWriteNbiOnThread, Ut_HcommWriteNbiOnThread_When_950_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::Write)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    res = HcommWriteNbiOnThread(thread, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtCpuHcommWriteNbiOnThread, Ut_HcommWriteNbiOnThread_When_950_Thread_IsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::Write)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    // thread is cast to void in the builtin branch — nullptr is acceptable.
    res = HcommWriteNbiOnThread(0, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtCpuHcommWriteNbiOnThread, Ut_HcommWriteNbiOnThread_When_Channel_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    // 950 下内置分支对通道指针判空：null 句柄返回 E_PTR
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    res = HcommWriteNbiOnThread(thread, 0, dst, src, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtCpuHcommWriteNbiOnThread, Ut_HcommWriteNbiOnThread_When_Src_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    // src is checked before hrtGetDeviceType, so no mock needed.
    res = HcommWriteNbiOnThread(thread, channel, dst, nullptr, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtCpuHcommWriteNbiOnThread, Ut_HcommWriteNbiOnThread_When_Dst_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    // dst is checked before hrtGetDeviceType, so no mock needed.
    res = HcommWriteNbiOnThread(thread, channel, nullptr, src, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtCpuHcommWriteNbiOnThread, Ut_HcommWriteNbiOnThread_When_950_Write_Fails_Expect_ErrorCodePropagated)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::Write)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    res = HcommWriteNbiOnThread(thread, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}

// 910C：非 950/960 设备上内置 NBI 写路径不支持

TEST_F(UtCpuHcommWriteNbiOnThread, Ut_HcommWriteNbiOnThread_When_910C_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t910C)).will(returnValue(HCCL_SUCCESS));
    res = HcommWriteNbiOnThread(thread, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

// 910C + null channel：devType 门控在通道判空之前，返回 E_NOT_SUPPORT 而非 E_PTR

TEST_F(UtCpuHcommWriteNbiOnThread, Ut_HcommWriteNbiOnThread_When_910C_Channel_IsNull_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t910C)).will(returnValue(HCCL_SUCCESS));
    res = HcommWriteNbiOnThread(thread, 0, dst, src, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}
