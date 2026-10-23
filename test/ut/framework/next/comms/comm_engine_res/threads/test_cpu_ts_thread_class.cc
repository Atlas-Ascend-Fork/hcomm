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
#include "local_notify_impl.h"
#include "llt_hccl_stub_rank_graph.h"

class TestCpuTsThread : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(TestCpuTsThread, Ut_CpuTsThread_Init_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Stream* stream = cpuThread.GetStream();
    EXPECT_NE(stream, nullptr);
    uint32_t notifyNum = cpuThread.GetNotifyNum();
    EXPECT_EQ(2, notifyNum);
    void* notify = cpuThread.GetNotify(1);
    EXPECT_NE(nullptr, notify);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_Init_When_GetRunSideIsDeviceFailed_Expect_Return_HCCL_E_DRV)
{
    MOCKER(GetRunSideIsDevice).stubs().will(returnValue(HCCL_E_DRV));
    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_Init_When_AllocDeviceStream_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_Init_When_AllocNotifyFailed_Expect_Return_HCCL_E_RUNTIME)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyGetOffset).stubs().will(returnValue(HCCL_E_RUNTIME));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_Init_On_A5_Host_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Stream* stream = cpuThread.GetStream();
    EXPECT_NE(stream, nullptr);
    uint32_t notifyNum = cpuThread.GetNotifyNum();
    EXPECT_EQ(2, notifyNum);
    void* notify = cpuThread.GetNotify(1);
    EXPECT_NE(nullptr, notify);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_Init_On_Device_When_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{true};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_Init_On_A3_Host_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Stream* stream = cpuThread.GetStream();
    EXPECT_NE(stream, nullptr);
    uint32_t notifyNum = cpuThread.GetNotifyNum();
    EXPECT_EQ(2, notifyNum);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetMaster_When_Inited_Expect_Return_False)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool isMaster = cpuThread.GetMaster();
    EXPECT_EQ(false, isMaster);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_SetIsMaster_When_Called_Expect_MasterIsTrue)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    cpuThread.SetIsMaster(true);
    bool isMaster = cpuThread.GetMaster();
    EXPECT_EQ(true, isMaster);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetStream_When_Inited_Expect_NotNull)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Stream* stream = cpuThread.GetStream();
    EXPECT_NE(nullptr, stream);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LaunchTask_When_Inited_Expect_NoThrow)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_NO_THROW(cpuThread.LaunchTask());
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetUniqueId_When_Inited_Expect_NotEmpty)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string& uniqueId = cpuThread.GetUniqueId();
    EXPECT_NE("", uniqueId);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetStreamLitePtr_When_Inited_Expect_Nullptr)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* streamLitePtr = cpuThread.GetStreamLitePtr();
    EXPECT_EQ(nullptr, streamLitePtr);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalCopy_When_NotA5Device_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 1024;

    ret = cpuThread.LocalCopy(dst, src, size);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalCopy_When_A5Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              return HCCL_SUCCESS;
          };
    cpuThread.SetAddTaskInfoCallback(callback);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 1024;

    MOCKER(hrtMemAsyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    ret = cpuThread.LocalCopy(dst, src, size);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalCopy_When_SizeIsZero_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 0;

    ret = cpuThread.LocalCopy(dst, src, size);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalCopy_When_SrcEqualsDst_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr = reinterpret_cast<void*>(0x1000);
    uint64_t size = 1024;

    ret = cpuThread.LocalCopy(ptr, ptr, size);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalCopy_When_OpFailed_Expect_Return_Error)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 1024;

    MOCKER(hrtMemAsyncCopy).stubs().will(returnValue(HCCL_E_INTERNAL));

    ret = cpuThread.LocalCopy(dst, src, size);
    EXPECT_EQ(HCCL_E_INTERNAL, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalReduce_When_NotA5Device_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 1024;

    ret = cpuThread.LocalReduce(dst, src, size, HcommDataType::HCOMM_DATA_TYPE_INT32, HcommReduceOp::HCOMM_REDUCE_SUM);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalReduce_When_A5Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              return HCCL_SUCCESS;
          };
    cpuThread.SetAddTaskInfoCallback(callback);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 1024;

    MOCKER(hrtReduceAsync).stubs().will(returnValue(HCCL_SUCCESS));

    ret = cpuThread.LocalReduce(dst, src, size, HcommDataType::HCOMM_DATA_TYPE_INT32, HcommReduceOp::HCOMM_REDUCE_SUM);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalReduce_When_InvalidDataType_Expect_Return_HCCL_E_PARA)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 1024;

    ret = cpuThread.LocalReduce(
        dst, src, size, HcommDataType::HCOMM_DATA_TYPE_RESERVED, HcommReduceOp::HCOMM_REDUCE_SUM);
    EXPECT_EQ(HCCL_E_PARA, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalReduce_When_InvalidReduceOp_Expect_Return_HCCL_E_PARA)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 1024;

    ret = cpuThread.LocalReduce(
        dst, src, size, HcommDataType::HCOMM_DATA_TYPE_INT32, HcommReduceOp::HCOMM_REDUCE_RESERVED);
    EXPECT_EQ(HCCL_E_PARA, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalReduce_When_OpFailed_Expect_Return_Error)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* dst = reinterpret_cast<void*>(0x1000);
    const void* src = reinterpret_cast<const void*>(0x2000);
    uint64_t size = 1024;

    MOCKER(hrtReduceAsync).stubs().will(returnValue(HCCL_E_INTERNAL));

    ret = cpuThread.LocalReduce(dst, src, size, HcommDataType::HCOMM_DATA_TYPE_INT32, HcommReduceOp::HCOMM_REDUCE_SUM);
    EXPECT_EQ(HCCL_E_INTERNAL, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalNotifyRecord_When_NotA5Device_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ThreadHandle dstThread = 0;
    uint32_t dstNotifyIdx = 0;

    ret = cpuThread.LocalNotifyRecord(dstThread, dstNotifyIdx);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalNotifyRecord_When_A5Device_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              return HCCL_SUCCESS;
          };
    cpuThread.SetAddTaskInfoCallback(callback);

    CpuTsThread dstThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    ret = dstThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ThreadHandle dstThreadHandle = reinterpret_cast<ThreadHandle>(&dstThread);
    uint32_t dstNotifyIdx = 0;

    ret = cpuThread.LocalNotifyRecord(dstThreadHandle, dstNotifyIdx);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalNotifyWait_When_NotA5Device_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t notifyIdx = 0;
    uint32_t timeOut = 1000;

    ret = cpuThread.LocalNotifyWait(notifyIdx, timeOut);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalNotifyWait_When_A5Device_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback
        = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
              return HCCL_SUCCESS;
          };
    cpuThread.SetAddTaskInfoCallback(callback);

    uint32_t notifyIdx = 0;
    uint32_t timeOut = 1000;

    ret = cpuThread.LocalNotifyWait(notifyIdx, timeOut);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalNotifyRecord_SingleThread_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t notifyId = 0;

    ret = cpuThread.LocalNotifyRecord(notifyId);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalNotifyWait_SingleThread_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t notifyId = 0;

    ret = cpuThread.LocalNotifyWait(notifyId);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_SupplementNotify_When_Normal_Expect_Return_HCCL_SUCCESS)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t notifyNum = cpuThread.GetNotifyNum();
    EXPECT_EQ(2, notifyNum);

    uint32_t addNotifyNum = 3;
    ret = cpuThread.SupplementNotify(addNotifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t newNotifyNum = cpuThread.GetNotifyNum();
    EXPECT_EQ(notifyNum + addNotifyNum, newNotifyNum);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_SupplementNotify_When_DeviceStreamType_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    ret = cpuThread.SupplementNotify(3);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_SupplementNotify_When_DeviceNotifyLoadType_Expect_Return_HCCL_E_NOT_SUPPORT)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    ret = cpuThread.SupplementNotify(3);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetNotify_When_OutOfRange_Expect_ReturnNullptr)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    LocalNotify* notify = cpuThread.GetNotify(2);
    EXPECT_EQ(notify, nullptr);

    notify = cpuThread.GetNotify(100);
    EXPECT_EQ(notify, nullptr);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_DeInit_When_Normal_Expect_ReturnSuccess)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = cpuThread.DeInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(cpuThread.GetNotifyNum(), 0u);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_TryLaunchTask_When_Normal_Expect_NoThrow)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_NO_THROW(cpuThread.TryLaunchTask());
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalNotifyRecord_SingleParam_When_A5_Expect_ReturnNotSupport)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = cpuThread.LocalNotifyRecord(0);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_LocalNotifyWait_SingleParam_When_A5_Expect_ReturnNotSupport)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = cpuThread.LocalNotifyWait(0);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetNotify_When_NotInited_Expect_Nullptr)
{
    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    LocalNotify* notify = cpuThread.GetNotify(0);
    EXPECT_EQ(nullptr, notify);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetNotify_When_InitedAndOutOfRange_Expect_Nullptr)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    LocalNotify* notify = cpuThread.GetNotify(cpuThread.GetNotifyNum());
    EXPECT_EQ(nullptr, notify);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetNotifySize_When_NotInited_Expect_Zero)
{
    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    size_t notifySize = cpuThread.GetNotifySize();
    EXPECT_EQ(0u, notifySize);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetNotifySize_When_Inited_Expect_EqualsNotifyNum)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    size_t notifySize = cpuThread.GetNotifySize();
    EXPECT_EQ(static_cast<size_t>(cpuThread.GetNotifyNum()), notifySize);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_FormatThreadInfo_When_NotInited_Expect_NoCrashAndEmptyNotifyIds)
{
    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    std::string info;
    EXPECT_NO_THROW(info = cpuThread.FormatThreadInfo());
    EXPECT_NE(info.find("notifyIds[]"), std::string::npos);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_FormatThreadInfo_When_Inited_Expect_ContainsStreamId)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CpuTsThread cpuThread(StreamType::STREAM_TYPE_ONLINE, 2, NotifyLoadType::HOST_NOTIFY);
    HcclResult ret = cpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string info = cpuThread.FormatThreadInfo();
    EXPECT_NE(info.find("streamId["), std::string::npos);
    EXPECT_NE(info.find("threadHandle[0x"), std::string::npos);
    EXPECT_NE(info.find("notifyIds["), std::string::npos);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetNotify_When_Injected_Expect_ReturnInjectedNotify)
{
    LocalNotify notify0;
    notify0.notifyId_ = 100;
    LocalNotify notify1;
    notify1.notifyId_ = 200;
    LocalNotify* injectedNotifys[2] = {&notify0, &notify1};

    CpuTsThread cpuThread(nullptr, injectedNotifys, 2);
    LocalNotify* notify0Ptr = cpuThread.GetNotify(0);
    LocalNotify* notify1Ptr = cpuThread.GetNotify(1);
    LocalNotify* notify2Ptr = cpuThread.GetNotify(2);
    EXPECT_EQ(&notify0, notify0Ptr);
    EXPECT_EQ(&notify1, notify1Ptr);
    EXPECT_EQ(nullptr, notify2Ptr);
    EXPECT_EQ(100u, notify0Ptr->notifyId_);
    EXPECT_EQ(200u, notify1Ptr->notifyId_);
}

TEST_F(TestCpuTsThread, Ut_CpuTsThread_GetNotifySize_When_Injected_Expect_InjectedSize)
{
    LocalNotify notify0;
    LocalNotify notify1;
    LocalNotify* injectedNotifys[2] = {&notify0, &notify1};

    CpuTsThread cpuThread(nullptr, injectedNotifys, 2);
    size_t notifySize = cpuThread.GetNotifySize();
    EXPECT_EQ(2u, notifySize);
}
